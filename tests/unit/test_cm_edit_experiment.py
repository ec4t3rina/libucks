"""CM-B Stage 1, step 4 — the experiment orchestrator and its gate.

Wires the pieces together: apply a controlled edit, re-derive teacher answers,
build ground truth with a full re-distill, run each repair method, and score
them against a gate fixed BEFORE the run.

Everything model-dependent goes through a `TrialRunner` protocol, so the
orchestration — trial matrix, ordering, resume, gate arithmetic — is testable
on CPU with a fake runner. That is deliberate: the gate is where a result gets
declared, and it is the last place that should only ever be exercised by a
six-hour GPU job.

The gate wording in docs/cm-b-plan.md is "within 10% of full-re-distill KL at
<=25% of the cost, on >=3 of the 4 edit types". "Within 10% of full-re-distill
KL" is ambiguous, so the resolution is pinned here explicitly:

    repaired_kl <= redistill_kl * 1.10        (quality)
    repair_seconds <= redistill_seconds*0.25  (cost)

where both KLs are each cartridge's own distillation KL against the teacher —
the same number CartridgeTrainer already reports, so the two are directly
comparable. KL(repaired || re-distilled) is recorded alongside as
`agreement_kl`, but is NOT the gate: it has no natural scale to threshold
against, whereas a ratio between two like-for-like KLs does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from libucks.cache_augmentation.cartridge_edit import RepairResult
from scripts.cm_edit_experiment import (
    ExperimentSpec,
    TrialKey,
    TrialResult,
    evaluate_gate,
    load_completed,
    trial_matrix,
    write_trial,
)


def _trial(
    edit_type: str = "constant",
    method: str = "slots",
    seed: int = 0,
    *,
    repaired_kl: float = 0.20,
    redistill_kl: float = 0.20,
    stale_kl: float = 0.90,
    repair_seconds: float = 600.0,
    redistill_seconds: float = 7200.0,
    n_changed: int = 12,
) -> TrialResult:
    return TrialResult(
        key=TrialKey(edit_type=edit_type, method=method, seed=seed),
        repair=RepairResult(
            method=method, seconds=repair_seconds, n_queries=n_changed,
            n_trainable_params=1000, final_kl=repaired_kl,
            kl_history=[stale_kl, repaired_kl],
        ),
        redistill_kl=redistill_kl,
        redistill_seconds=redistill_seconds,
        stale_kl=stale_kl,
        agreement_kl=0.01,
        grounding_repaired=5,
        grounding_redistilled=5,
        n_fixtures=8,
    )


# ---------------------------------------------------------------------------
# Trial matrix
# ---------------------------------------------------------------------------

class TestTrialMatrix:
    def test_covers_every_edit_type_method_and_seed(self):
        spec = ExperimentSpec(
            repo=Path("/r"), source_file="a.py", bucket_id="b1",
            edit_types=("rename", "constant"),
            methods=("continue", "slots"),
            seeds=(0, 1),
        )
        keys = trial_matrix(spec)
        assert len(keys) == 2 * 2 * 2
        assert len(set(keys)) == len(keys), "duplicate trial keys"

    def test_is_ordered_so_all_methods_share_one_ground_truth(self):
        """A full re-distill costs ~7,200 s. Running it once per (edit, seed)
        and reusing it across methods is the difference between a 3-hour and a
        9-hour experiment — so trials must be grouped, not interleaved."""
        spec = ExperimentSpec(
            repo=Path("/r"), source_file="a.py", bucket_id="b1",
            edit_types=("rename", "constant"),
            methods=("continue", "slots", "lowrank"),
            seeds=(0, 1),
        )
        groups = [(k.edit_type, k.seed) for k in trial_matrix(spec)]
        # every (edit_type, seed) block must be contiguous
        first_seen = {}
        for i, g in enumerate(groups):
            if g in first_seen:
                assert groups[i - 1] == g, f"group {g} is not contiguous"
            else:
                first_seen[g] = i

    def test_is_deterministic(self):
        spec = ExperimentSpec(
            repo=Path("/r"), source_file="a.py", bucket_id="b1",
            edit_types=("rename", "constant"), methods=("slots",), seeds=(0, 1),
        )
        assert trial_matrix(spec) == trial_matrix(spec)

    def test_rejects_an_unknown_repair_method(self):
        with pytest.raises(ValueError, match="unknown repair method"):
            ExperimentSpec(
                repo=Path("/r"), source_file="a.py", bucket_id="b1",
                edit_types=("rename",), methods=("telepathy",), seeds=(0,),
            )

    def test_rejects_an_unknown_edit_type(self):
        with pytest.raises(ValueError, match="unknown edit type"):
            ExperimentSpec(
                repo=Path("/r"), source_file="a.py", bucket_id="b1",
                edit_types=("vibes",), methods=("slots",), seeds=(0,),
            )

    def test_rejects_an_empty_seed_list(self):
        with pytest.raises(ValueError, match="at least one seed"):
            ExperimentSpec(
                repo=Path("/r"), source_file="a.py", bucket_id="b1",
                edit_types=("rename",), methods=("slots",), seeds=(),
            )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

class TestGateArithmetic:
    def test_passing_trial_meets_both_quality_and_cost(self):
        v = evaluate_gate([_trial(repaired_kl=0.21, redistill_kl=0.20,
                                  repair_seconds=1000.0)])
        assert v.per_trial[0].passed

    def test_kl_exactly_at_the_ten_percent_boundary_passes(self):
        v = evaluate_gate([_trial(repaired_kl=0.22, redistill_kl=0.20)])
        assert v.per_trial[0].passed

    def test_kl_just_past_the_boundary_fails(self):
        v = evaluate_gate([_trial(repaired_kl=0.2201, redistill_kl=0.20)])
        assert not v.per_trial[0].passed
        assert "quality" in v.per_trial[0].reason

    def test_cost_exactly_at_the_ceiling_passes(self):
        v = evaluate_gate([_trial(repair_seconds=1800.0, redistill_seconds=7200.0)])
        assert v.per_trial[0].passed

    def test_cost_just_past_the_ceiling_fails(self):
        v = evaluate_gate([_trial(repair_seconds=1801.0, redistill_seconds=7200.0)])
        assert not v.per_trial[0].passed
        assert "cost" in v.per_trial[0].reason

    def test_a_repair_better_than_redistill_still_passes(self):
        """Cheap repair beating the baseline is a pass, not an anomaly."""
        v = evaluate_gate([_trial(repaired_kl=0.05, redistill_kl=0.20)])
        assert v.per_trial[0].passed


class TestGateRefusesUninformativeTrials:
    """The pre-committed honest outcome: if the edit barely disturbed the
    cartridge, easy repair is insensitivity, not a mechanism."""

    def test_zero_changed_queries_cannot_pass(self):
        v = evaluate_gate([_trial(n_changed=0, repair_seconds=1.0)])
        assert not v.per_trial[0].passed
        assert "uninformative" in v.per_trial[0].reason.lower()

    def test_staleness_floor_at_the_redistill_level_cannot_pass(self):
        """If the STALE cartridge already matches the re-distilled one, the
        edit did nothing and the repair had nothing to do."""
        v = evaluate_gate([_trial(stale_kl=0.205, redistill_kl=0.20)])
        assert not v.per_trial[0].passed
        assert "staleness floor" in v.per_trial[0].reason.lower()

    def test_a_real_disturbance_is_accepted(self):
        v = evaluate_gate([_trial(stale_kl=0.90, redistill_kl=0.20)])
        assert v.per_trial[0].passed

    def test_uninformative_trials_are_counted_separately(self):
        v = evaluate_gate([_trial(edit_type="rename", n_changed=0),
                           _trial(edit_type="constant")])
        assert v.n_uninformative == 1


class TestGateVerdict:
    def _four_types(self, passing: int) -> list[TrialResult]:
        types = ["rename", "constant", "branch", "delete"]
        out = []
        for i, t in enumerate(types):
            ok = i < passing
            out.append(_trial(edit_type=t, repaired_kl=0.20 if ok else 9.0))
        return out

    @pytest.mark.parametrize("n_pass,expected", [(4, True), (3, True),
                                                 (2, False), (0, False)])
    def test_needs_three_of_four_edit_types(self, n_pass: int, expected: bool):
        v = evaluate_gate(self._four_types(n_pass), method="slots")
        assert v.passed is expected

    def test_verdict_is_per_method(self):
        """Each method gets its own verdict; 'slots' passing is the result,
        'continue' passing is only the baseline."""
        trials = [_trial(edit_type=t, method="continue") for t in
                  ("rename", "constant", "branch", "delete")]
        trials += [_trial(edit_type=t, method="slots", repaired_kl=9.0) for t in
                   ("rename", "constant", "branch", "delete")]
        assert evaluate_gate(trials, method="continue").passed
        assert not evaluate_gate(trials, method="slots").passed

    def test_two_seeds_of_one_edit_type_count_once(self):
        """Four passing seeds of ONE edit type is not four edit types."""
        trials = [_trial(edit_type="rename", seed=s) for s in range(4)]
        v = evaluate_gate(trials, method="slots")
        assert v.n_edit_types_passed == 1
        assert not v.passed

    def test_an_edit_type_passes_only_if_most_of_its_seeds_do(self):
        trials = [_trial(edit_type="rename", seed=0, repaired_kl=0.20),
                  _trial(edit_type="rename", seed=1, repaired_kl=9.0),
                  _trial(edit_type="rename", seed=2, repaired_kl=9.0)]
        assert evaluate_gate(trials, method="slots").n_edit_types_passed == 0

    def test_two_seeds_require_both_to_pass(self):
        """SEED_MAJORITY is a STRICT majority, so 1-of-2 is 0.5 and fails.

        This changes experiment design, not just arithmetic: two seeds is a
        harsher bar than three, where 2-of-3 suffices. Recorded here so the
        choice of seed count is deliberate rather than a cost accident.
        """
        split = [_trial(edit_type="rename", seed=0, repaired_kl=0.20),
                 _trial(edit_type="rename", seed=1, repaired_kl=9.0)]
        assert evaluate_gate(split, method="slots").n_edit_types_passed == 0

        both = [_trial(edit_type="rename", seed=s, repaired_kl=0.20)
                for s in (0, 1)]
        assert evaluate_gate(both, method="slots").n_edit_types_passed == 1

    def test_three_seeds_allow_one_failure(self):
        trials = [_trial(edit_type="rename", seed=0, repaired_kl=0.20),
                  _trial(edit_type="rename", seed=1, repaired_kl=0.20),
                  _trial(edit_type="rename", seed=2, repaired_kl=9.0)]
        assert evaluate_gate(trials, method="slots").n_edit_types_passed == 1

    def test_empty_results_do_not_pass(self):
        v = evaluate_gate([], method="slots")
        assert not v.passed
        assert v.n_edit_types_passed == 0

    def test_summary_mentions_the_thresholds_actually_used(self):
        v = evaluate_gate(self._four_types(4), method="slots")
        assert "10" in v.summary and "25" in v.summary


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class FakeRunner:
    """Records what the orchestrator asked for, in order."""

    def __init__(self, *, answers_change: bool = True) -> None:
        self.calls: list[str] = []
        self.redistills = 0
        self.repairs: list[tuple[str, tuple[str, ...]]] = []
        self._answers_change = answers_change
        self._edit_applied = False

    def queries(self) -> list[str]:
        self.calls.append("queries")
        return ["q1", "q2", "q3"]

    def teacher_answers(self, queries):
        self.calls.append("teacher_answers")
        if self._edit_applied and self._answers_change:
            return {"q1": "NEW", "q2": "same", "q3": "same"}
        return {q: "same" if q != "q1" else "OLD" for q in queries}

    def full_redistill(self):
        self.calls.append("full_redistill")
        self.redistills += 1
        return 0.20, 7200.0

    def stale_score(self) -> float:
        self.calls.append("stale_score")
        return 0.90

    def repair(self, method, changed):
        self.calls.append(f"repair:{method}")
        self.repairs.append((method, tuple(changed)))
        return RepairResult(method=method, seconds=600.0, n_queries=len(changed),
                            n_trainable_params=10, final_kl=0.20,
                            kl_history=[0.9, 0.2])

    def agreement(self) -> float:
        return 0.01

    def grounding(self):
        return 5, 5, 8


class RecordingEdits:
    """Stands in for cm_make_edits so no real repo is touched."""

    def __init__(self, runner: FakeRunner) -> None:
        self.runner = runner
        self.applied: list[tuple[str, int]] = []
        self.reverted: list[tuple[str, int]] = []

    def apply(self, edit_type: str, seed: int) -> None:
        self.applied.append((edit_type, seed))
        self.runner._edit_applied = True

    def revert(self, edit_type: str, seed: int) -> None:
        self.reverted.append((edit_type, seed))
        self.runner._edit_applied = False


@pytest.fixture
def spec_2x2() -> ExperimentSpec:
    return ExperimentSpec(
        repo=Path("/r"), source_file="a.py", bucket_id="b1",
        edit_types=("rename", "constant"),
        methods=("continue", "slots"),
        seeds=(0,),
    )


class TestRunExperiment:
    def test_ground_truth_is_computed_once_per_edit_seed_group(
        self, tmp_path: Path, spec_2x2
    ):
        """Re-distilling per METHOD instead of per group triples an
        already-multi-hour run for nothing."""
        from scripts.cm_edit_experiment import run_experiment

        runner = FakeRunner()
        edits = RecordingEdits(runner)
        run_experiment(spec_2x2, runner, tmp_path / "t.jsonl", edits=edits)

        assert runner.redistills == 2, (
            f"2 groups x 2 methods must re-distill twice, not {runner.redistills}"
        )

    def test_every_trial_is_recorded(self, tmp_path: Path, spec_2x2):
        from scripts.cm_edit_experiment import run_experiment

        out = tmp_path / "t.jsonl"
        runner = FakeRunner()
        results = run_experiment(spec_2x2, runner, out,
                                 edits=RecordingEdits(runner))
        assert len(results) == 4
        assert len(out.read_text().strip().splitlines()) == 4

    def test_edit_is_applied_before_and_reverted_after_each_group(
        self, tmp_path: Path, spec_2x2
    ):
        """Every method must start from an identical tree, or the comparison
        between them is meaningless."""
        from scripts.cm_edit_experiment import run_experiment

        runner = FakeRunner()
        edits = RecordingEdits(runner)
        run_experiment(spec_2x2, runner, tmp_path / "t.jsonl", edits=edits)

        assert edits.applied == [("rename", 0), ("constant", 0)]
        assert edits.reverted == edits.applied, "every edit must be reverted"

    def test_only_changed_queries_reach_the_repair(self, tmp_path: Path, spec_2x2):
        """Retraining on unchanged queries spends the budget being measured."""
        from scripts.cm_edit_experiment import run_experiment

        runner = FakeRunner()
        run_experiment(spec_2x2, runner, tmp_path / "t.jsonl",
                       edits=RecordingEdits(runner))

        assert runner.repairs, "no repair was run"
        for method, changed in runner.repairs:
            assert changed == ("q1",), f"{method} got {changed}, expected only q1"

    def test_teacher_answers_are_taken_before_and_after_the_edit(
        self, tmp_path: Path, spec_2x2
    ):
        from scripts.cm_edit_experiment import run_experiment

        runner = FakeRunner()
        run_experiment(spec_2x2, runner, tmp_path / "t.jsonl",
                       edits=RecordingEdits(runner))
        assert runner.calls.count("teacher_answers") == 2 * 2  # 2 groups x (pre, post)

    def test_an_edit_that_changes_nothing_is_skipped_loudly(
        self, tmp_path: Path, spec_2x2
    ):
        """NoChangeDetected must abort that group, not silently record a
        zero-query 'success'."""
        from scripts.cm_edit_experiment import run_experiment

        runner = FakeRunner(answers_change=False)
        results = run_experiment(spec_2x2, runner, tmp_path / "t.jsonl",
                                 edits=RecordingEdits(runner))
        assert results == []
        assert runner.repairs == [], "must not repair when nothing changed"

    def test_resume_skips_recorded_trials(self, tmp_path: Path, spec_2x2):
        from scripts.cm_edit_experiment import run_experiment

        out = tmp_path / "t.jsonl"
        write_trial(out, _trial(edit_type="rename", method="continue", seed=0))

        runner = FakeRunner()
        results = run_experiment(spec_2x2, runner, out,
                                 edits=RecordingEdits(runner))
        keys = {(r.key.edit_type, r.key.method) for r in results}
        assert ("rename", "continue") not in keys
        assert len(results) == 3

    def test_a_fully_completed_matrix_does_no_work(self, tmp_path: Path, spec_2x2):
        from scripts.cm_edit_experiment import run_experiment

        out = tmp_path / "t.jsonl"
        for e in ("rename", "constant"):
            for m in ("continue", "slots"):
                write_trial(out, _trial(edit_type=e, method=m, seed=0))

        runner = FakeRunner()
        assert run_experiment(spec_2x2, runner, out,
                              edits=RecordingEdits(runner)) == []
        assert runner.redistills == 0, "resume must not re-run ground truth"


class TestResume:
    def test_written_trials_are_read_back(self, tmp_path: Path):
        out = tmp_path / "trials.jsonl"
        write_trial(out, _trial(edit_type="rename", method="slots", seed=0))
        write_trial(out, _trial(edit_type="constant", method="slots", seed=1))
        done = load_completed(out)
        assert TrialKey("rename", "slots", 0) in done
        assert TrialKey("constant", "slots", 1) in done

    def test_missing_file_means_nothing_completed(self, tmp_path: Path):
        assert load_completed(tmp_path / "absent.jsonl") == set()

    def test_a_truncated_final_line_is_skipped_not_fatal(self, tmp_path: Path):
        """A kill mid-write must not make the whole run unresumable — this
        project has lost overnight jobs to exactly that class of thing."""
        out = tmp_path / "trials.jsonl"
        write_trial(out, _trial(edit_type="rename", method="slots", seed=0))
        with out.open("a") as f:
            f.write('{"key": {"edit_type": "constant", "meth')
        done = load_completed(out)
        assert done == {TrialKey("rename", "slots", 0)}

    def test_each_trial_is_one_line(self, tmp_path: Path):
        out = tmp_path / "trials.jsonl"
        for i in range(3):
            write_trial(out, _trial(seed=i))
        assert len(out.read_text().strip().splitlines()) == 3

    def test_round_trip_preserves_the_numbers(self, tmp_path: Path):
        out = tmp_path / "trials.jsonl"
        t = _trial(repaired_kl=0.1234, repair_seconds=567.8)
        write_trial(out, t)
        row = json.loads(out.read_text().strip())
        assert row["repair"]["final_kl"] == pytest.approx(0.1234)
        assert row["repair"]["seconds"] == pytest.approx(567.8)
