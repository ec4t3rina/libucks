"""CM-B Stage 1, step 4 — orchestrate edit -> repair -> measure.

For each (edit type, seed):

    1. Apply a controlled edit and commit it        (cm_make_edits)
    2. Re-derive teacher answers; keep the queries whose answer moved
    3. GROUND TRUTH: full re-distill of the changed bucket  (~7,200 s)
    4. STALENESS FLOOR: score the un-repaired cartridge
    5. For each repair method: repair, then score          (cartridge_edit)
    6. Revert the edit so the next trial starts from an identical tree

Everything model-dependent goes through the `TrialRunner` protocol below, so
the orchestration — trial matrix, ordering, resume, gate arithmetic — is
CPU-testable with a fake. The gate is where a result gets declared; it should
not be exercised only by a six-hour GPU job.

Cost note driving the ordering: step 3 dominates everything else, so the trial
matrix groups all methods under one (edit type, seed) and the runner computes
ground truth once per group. Interleaving methods would re-distil per method
and roughly triple the wall clock.

Resume: every finished trial is appended to a JSONL as one line, and a re-run
skips whatever is already in it. Stage 0b lost a bucket to a wedge because a
checkpoint was write-only; a multi-hour matrix must not repeat that.

    uv run python scripts/cm_edit_experiment.py --repo <path> --file <rel.py> \
        --bucket <id> [--methods slots,continue] [--seeds 0,1] [--dry-run]
"""
from __future__ import annotations

# MPS allocator cap must be set before torch is imported anywhere, including
# transitively via libucks. Setting it afterwards is silently a no-op and has
# cost this project entire nights.
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from libucks.cache_augmentation.cartridge_edit import REPAIR_METHODS, RepairResult

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cm_make_edits import EDIT_TYPES  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered gate. Fixed before the run, per docs/cm-b-plan.md house rules.
# ---------------------------------------------------------------------------

KL_TOLERANCE = 0.10       # repaired_kl <= redistill_kl * (1 + this)
COST_CEILING = 0.25       # repair_seconds <= redistill_seconds * this
MIN_EDIT_TYPES = 3        # of the four
SEED_MAJORITY = 0.5       # STRICT: an edit type needs > this share of its seeds

# Note the strictness interacts with seed count, and that changes the bar, not
# just the cost. With 2 seeds, 1 pass is exactly 0.5 and therefore FAILS — both
# must pass. With 3 seeds, 2-of-3 suffices. Ground truth is ~7,200 s per
# (edit type, seed), so 4 types x 3 seeds is ~24 h of re-distills before any
# repair runs; 4 x 2 is ~16 h but is the harsher gate. Pick deliberately.

# A trial is only informative if the edit actually moved the cartridge. If the
# STALE cartridge is already within this factor of the re-distilled one, the
# edit disturbed nothing and any "repair" was free by construction.
MIN_STALENESS_GAP = 0.25  # stale_kl must exceed redistill_kl * (1 + this)


def _log(msg: str) -> None:
    print(f"[cm_edit_exp] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class TrialKey:
    edit_type: str
    method: str
    seed: int


@dataclass
class TrialResult:
    """One (edit type, method, seed) measurement.

    `redistill_kl` and `repair.final_kl` are each cartridge's own distillation
    KL against the teacher — the number CartridgeTrainer already reports — so
    they are directly comparable and a ratio between them is meaningful.

    `agreement_kl` is KL(repaired || re-distilled), recorded because the plan
    names it the primary metric and it is free of decode-loop variance. It is
    NOT the gate: it has no natural scale to threshold against, whereas a ratio
    between two like-for-like KLs does.
    """

    key: TrialKey
    repair: RepairResult
    redistill_kl: float
    redistill_seconds: float
    stale_kl: float
    agreement_kl: float
    grounding_repaired: int = 0
    grounding_redistilled: int = 0
    n_fixtures: int = 0

    @property
    def cost_ratio(self) -> float:
        return self.repair.cost_ratio(self.redistill_seconds)

    @property
    def kl_ratio(self) -> float:
        """repaired / re-distilled. <= 1 + KL_TOLERANCE is the quality bar."""
        if self.redistill_kl <= 0:
            return float("inf") if self.repair.final_kl > 0 else 1.0
        return self.repair.final_kl / self.redistill_kl


@dataclass
class TrialVerdict:
    key: TrialKey
    passed: bool
    reason: str


@dataclass
class GateVerdict:
    method: str
    passed: bool
    n_edit_types_passed: int
    n_uninformative: int
    per_trial: list[TrialVerdict] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExperimentSpec:
    repo: Path
    source_file: str
    bucket_id: str
    edit_types: tuple[str, ...] = EDIT_TYPES
    methods: tuple[str, ...] = REPAIR_METHODS
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 4
    lr: float = 1e-2

    def __post_init__(self) -> None:
        bad_m = [m for m in self.methods if m not in REPAIR_METHODS]
        if bad_m:
            raise ValueError(
                f"unknown repair method(s) {bad_m}; expected {list(REPAIR_METHODS)}"
            )
        bad_e = [e for e in self.edit_types if e not in EDIT_TYPES]
        if bad_e:
            raise ValueError(
                f"unknown edit type(s) {bad_e}; expected {list(EDIT_TYPES)}"
            )
        if not self.seeds:
            raise ValueError("need at least one seed")
        if not self.edit_types:
            raise ValueError("need at least one edit type")
        if not self.methods:
            raise ValueError("need at least one repair method")


# ---------------------------------------------------------------------------
# Trial matrix
# ---------------------------------------------------------------------------

def trial_matrix(spec: ExperimentSpec) -> list[TrialKey]:
    """All trials, grouped so every method shares one ground-truth re-distill.

    Order is (edit_type, seed) outer, method inner. The full re-distill in
    step 3 costs ~7,200 s and dominates; computing it once per group instead of
    once per method is the difference between a 3-hour and a 9-hour run.
    """
    return [
        TrialKey(edit_type=e, method=m, seed=s)
        for e in spec.edit_types
        for s in spec.seeds
        for m in spec.methods
    ]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _judge(t: TrialResult) -> TrialVerdict:
    # Informativeness first: a trial that measured nothing cannot pass, however
    # good its numbers look. This is the pre-committed honest outcome — easy
    # repair because the edit did nothing is insensitivity, not a mechanism.
    if t.repair.n_queries == 0:
        return TrialVerdict(t.key, False,
                            "uninformative: no teacher answer changed, so the "
                            "repair was free by construction")

    if t.stale_kl <= t.redistill_kl * (1 + MIN_STALENESS_GAP):
        return TrialVerdict(
            t.key, False,
            f"uninformative: staleness floor {t.stale_kl:.3f} is within "
            f"{MIN_STALENESS_GAP:.0%} of the re-distill {t.redistill_kl:.3f} — "
            "the edit did not disturb the cartridge"
        )

    quality_ok = t.repair.final_kl <= t.redistill_kl * (1 + KL_TOLERANCE)
    cost_ok = t.repair.seconds <= t.redistill_seconds * COST_CEILING

    if not quality_ok and not cost_ok:
        return TrialVerdict(t.key, False,
                            f"quality and cost: kl_ratio={t.kl_ratio:.3f}, "
                            f"cost_ratio={t.cost_ratio:.3f}")
    if not quality_ok:
        return TrialVerdict(t.key, False, f"quality: kl_ratio={t.kl_ratio:.3f} "
                                          f"> {1 + KL_TOLERANCE:.2f}")
    if not cost_ok:
        return TrialVerdict(t.key, False, f"cost: cost_ratio={t.cost_ratio:.3f} "
                                          f"> {COST_CEILING:.2f}")
    return TrialVerdict(t.key, True,
                        f"kl_ratio={t.kl_ratio:.3f}, cost_ratio={t.cost_ratio:.3f}")


def evaluate_gate(
    results: Iterable[TrialResult],
    method: str | None = None,
) -> GateVerdict:
    """Score one repair method against the pre-registered gate.

    An edit TYPE counts as passed when more than SEED_MAJORITY of its seeds
    pass, so a single lucky seed cannot carry it. The overall verdict needs
    MIN_EDIT_TYPES distinct types — four passing seeds of one edit type is one
    edit type, not four.
    """
    rows = [t for t in results if method is None or t.key.method == method]
    verdicts = [_judge(t) for t in rows]
    by_key = {v.key: v for v in verdicts}

    by_type: dict[str, list[bool]] = {}
    for t in rows:
        by_type.setdefault(t.key.edit_type, []).append(by_key[t.key].passed)

    n_types_passed = sum(
        1 for oks in by_type.values()
        if oks and (sum(oks) / len(oks)) > SEED_MAJORITY
    )
    n_uninformative = sum(1 for v in verdicts if "uninformative" in v.reason)
    passed = n_types_passed >= MIN_EDIT_TYPES

    resolved = method or (rows[0].key.method if rows else "?")
    summary = (
        f"method={resolved}: {n_types_passed}/{len(by_type) or 0} edit types "
        f"passed (need {MIN_EDIT_TYPES}). Gate: repaired KL within "
        f"{KL_TOLERANCE:.0%} of full re-distill at <= {COST_CEILING:.0%} of its "
        f"cost. {n_uninformative} trial(s) uninformative."
    )
    return GateVerdict(
        method=resolved, passed=passed, n_edit_types_passed=n_types_passed,
        n_uninformative=n_uninformative, per_trial=verdicts, summary=summary,
    )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def write_trial(path: Path, result: TrialResult) -> None:
    """Append one trial as one JSON line, flushed immediately.

    One line per trial so a kill can only ever corrupt the last one, and
    load_completed tolerates exactly that.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed(path: Path) -> set[TrialKey]:
    """Trial keys already recorded. A truncated final line is skipped.

    Being killed mid-write must not make an entire multi-hour matrix
    unresumable — Stage 0b already lost a bucket to a write-only checkpoint.
    """
    path = Path(path)
    if not path.exists():
        return set()

    done: set[TrialKey] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            k = json.loads(line)["key"]
            done.add(TrialKey(k["edit_type"], k["method"], int(k["seed"])))
        except Exception:
            _log(f"skipping unreadable trial line ({len(line)} chars) — "
                 "likely a kill mid-write; it will be re-run")
    return done


# ---------------------------------------------------------------------------
# Runner protocol — everything that needs a model
# ---------------------------------------------------------------------------

class TrialRunner(Protocol):
    """The model-dependent half, injected so orchestration stays testable."""

    def queries(self) -> list[str]:
        """Self-study queries for the bucket under test."""

    def teacher_answers(self, queries: Sequence[str]) -> dict[str, str]:
        """Greedy teacher answers for the CURRENT state of the repo."""

    def full_redistill(self) -> tuple[float, float]:
        """Ground truth. Returns (final_kl, seconds)."""

    def stale_score(self) -> float:
        """Distillation KL of the un-repaired cartridge. The staleness floor."""

    def repair(self, method: str, changed: Sequence[str]) -> RepairResult:
        """Run one repair method over the changed queries."""

    def agreement(self) -> float:
        """KL(repaired || re-distilled) for the most recent repair."""

    def grounding(self) -> tuple[int, int, int]:
        """(repaired_hits, redistilled_hits, n_fixtures)."""


class EditDriver(Protocol):
    """Applying and reverting edits, injected so tests touch no real repo."""

    def apply(self, edit_type: str, seed: int) -> None: ...
    def revert(self, edit_type: str, seed: int) -> None: ...


class GitEditDriver:
    """Real edits via cm_make_edits, committed so libucks can see them.

    Committing matters: BucketKVCache keys freshness on (chunk_id, git_sha), so
    an uncommitted edit leaves the cartridge looking valid and the whole trial
    measures nothing.
    """

    def __init__(self, repo: Path, source_file: str) -> None:
        self._repo = Path(repo)
        self._file = source_file
        self._plans: dict[tuple[str, int], object] = {}

    def apply(self, edit_type: str, seed: int) -> None:
        from scripts.cm_make_edits import apply_edit, commit_edit, plan_edit

        plan = plan_edit(self._repo / self._file, edit_type, seed=seed)
        apply_edit(plan)
        commit_edit(plan, self._repo)
        self._plans[(edit_type, seed)] = plan

    def revert(self, edit_type: str, seed: int) -> None:
        from scripts.cm_make_edits import commit_edit, revert_edit

        plan = self._plans.pop((edit_type, seed), None)
        if plan is None:
            _log(f"WARNING: no plan recorded for {edit_type}/seed{seed}; "
                 "the tree may still carry the edit")
            return
        revert_edit(plan)
        commit_edit(plan, self._repo)


def run_experiment(
    spec: ExperimentSpec,
    runner: TrialRunner,
    out_path: Path,
    *,
    edits: EditDriver | None = None,
    resume: bool = True,
) -> list[TrialResult]:
    """Drive the matrix, one (edit type, seed) group at a time.

    Per group: apply the edit, diff teacher answers to find what actually
    moved, build ground truth ONCE, then run every repair method against it,
    and finally revert so the next group starts from an identical tree.

    A group whose edit moves no teacher answer is skipped entirely rather than
    recorded — a zero-query trial would otherwise sit in the results file
    looking like a very cheap success.
    """
    from libucks.cache_augmentation.cartridge_edit import (
        NoChangeDetected,
        changed_queries,
    )

    if edits is None:
        edits = GitEditDriver(spec.repo, spec.source_file)

    done = load_completed(out_path) if resume else set()
    if done:
        _log(f"resuming: {len(done)} trial(s) already complete")

    # Group the matrix so ground truth is computed once per (edit type, seed).
    groups: dict[tuple[str, int], list[TrialKey]] = {}
    for key in trial_matrix(spec):
        groups.setdefault((key.edit_type, key.seed), []).append(key)

    results: list[TrialResult] = []
    for (edit_type, seed), keys in groups.items():
        pending = [k for k in keys if k not in done]
        if not pending:
            _log(f"skip group {edit_type}/seed{seed} — all methods done")
            continue

        queries = runner.queries()
        before = runner.teacher_answers(queries)

        edits.apply(edit_type, seed)
        try:
            after = runner.teacher_answers(queries)
            try:
                changed = changed_queries(before, after, require_change=True)
            except NoChangeDetected as exc:
                _log(f"SKIP group {edit_type}/seed{seed}: {exc}")
                continue

            _log(f"{edit_type}/seed{seed}: {len(changed)}/{len(queries)} "
                 "teacher answers moved")
            _log("  ground truth (full re-distill, the slow one) ...")
            redistill_kl, redistill_seconds = runner.full_redistill()
            stale_kl = runner.stale_score()

            for key in pending:
                repair = runner.repair(key.method, changed)
                g_rep, g_re, n_fix = runner.grounding()
                result = TrialResult(
                    key=key, repair=repair,
                    redistill_kl=redistill_kl,
                    redistill_seconds=redistill_seconds,
                    stale_kl=stale_kl,
                    agreement_kl=runner.agreement(),
                    grounding_repaired=g_rep, grounding_redistilled=g_re,
                    n_fixtures=n_fix,
                )
                write_trial(out_path, result)
                results.append(result)
                _log(f"  {key.method}: kl_ratio={result.kl_ratio:.3f} "
                     f"cost_ratio={result.cost_ratio:.3f}")
        finally:
            # Always revert, even if a repair raised — otherwise the next group
            # stacks its edit on top of this one and every later trial is junk.
            edits.revert(edit_type, seed)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--file", required=True, help="source file to edit, relative to --repo")
    ap.add_argument("--bucket", required=True, help="bucket id owning that file")
    ap.add_argument("--edit-types", default=",".join(EDIT_TYPES))
    ap.add_argument("--methods", default=",".join(REPAIR_METHODS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=Path("tests/eval/results/cm/edit_experiment.jsonl"))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the trial matrix and cost estimate, run nothing")
    args = ap.parse_args()

    try:
        spec = ExperimentSpec(
            repo=args.repo, source_file=args.file, bucket_id=args.bucket,
            edit_types=tuple(args.edit_types.split(",")),
            methods=tuple(args.methods.split(",")),
            seeds=tuple(int(s) for s in args.seeds.split(",")),
            epochs=args.epochs,
        )
    except ValueError as exc:
        _log(f"ERROR: {exc}")
        return 1

    matrix = trial_matrix(spec)
    n_groups = len(spec.edit_types) * len(spec.seeds)
    _log(f"{len(matrix)} trials over {n_groups} ground-truth re-distills")
    _log(f"  edit types : {list(spec.edit_types)}")
    _log(f"  methods    : {list(spec.methods)}")
    _log(f"  seeds      : {list(spec.seeds)}")
    # 7,199 s/bucket measured in CM-A.2; repairs are the thing being measured,
    # so only the ground-truth cost is predictable in advance.
    _log(f"  ground truth alone ~= {n_groups * 7200 / 3600:.1f} h")

    done = set() if args.no_resume else load_completed(args.out)
    if done:
        _log(f"  {len(done)} already complete, {len(matrix) - len(done)} remaining")

    if args.dry_run:
        for k in matrix:
            mark = "done" if k in done else "todo"
            _log(f"  [{mark}] {k.edit_type}/{k.method}/seed{k.seed}")
        return 0

    _log("ERROR: the model-backed TrialRunner is not implemented yet. "
         "Stage 1 steps 1-3 (cm_make_edits, cartridge_edit) and this "
         "orchestration are ready; wiring CartridgeTrainer in is the next step "
         "and needs the GPU free.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
