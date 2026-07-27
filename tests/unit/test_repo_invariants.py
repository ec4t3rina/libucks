"""Structural invariants — the CM-B bug sweeps, made executable.

Four manual sweeps found the same three shapes over and over:

  * silent defaults that flip semantics   (mitosis vs merge token limit)
  * duplicated logic that drifts          (_encode_centroid x6, _read_chunk_content,
                                           four teacher-Q&A call sites, one raised)
  * manual cross-file invariants          (chunk_retriever escaping a nested scope)

A sweep only protects the repo on the day it is run. These tests re-run every
one of those checks on every commit, so a NEW instance fails here instead of
being found by the fifth manual pass — or not at all.

Each check carries an explicit allowlist of KNOWN, ACCEPTED exceptions with the
reason recorded. The tests are therefore about *change*: they pass today and
fail the moment something new appears. Do not silence one by extending an
allowlist without writing down why.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "libucks"


def _modname(p: Path) -> str:
    parts = list(p.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _pkg_files() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


def _parse(p: Path) -> ast.Module:
    return ast.parse(p.read_text())


def _import_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return []


# ---------------------------------------------------------------------------
# 1. Import graph health
# ---------------------------------------------------------------------------

def _runtime_graph() -> dict[str, set[str]]:
    """Edges executed at import time — the only ones that can ImportError.

    Imports inside `if TYPE_CHECKING:` or inside a function body are excluded;
    both are standard cycle-breakers and several are load-bearing here
    (mitosis imports Librarian function-locally on purpose).
    """
    mods = {_modname(p): p for p in _pkg_files()}
    graph: dict[str, set[str]] = defaultdict(set)
    for mod, p in mods.items():
        tree = _parse(p)
        lazy: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.If):
                t = n.test
                if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                    isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
                ):
                    lazy |= set(range(n.lineno, n.end_lineno + 1))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lazy |= set(range(n.lineno, n.end_lineno + 1))
        for n in ast.walk(tree):
            if n.__class__ not in (ast.Import, ast.ImportFrom):
                continue
            if n.lineno in lazy:
                continue
            for t in _import_targets(n):
                if t in mods and t != mod:
                    graph[mod].add(t)
    return graph


class TestImportGraph:
    def test_no_runtime_import_cycles(self):
        graph = _runtime_graph()
        cycles, state, stack = [], {}, []

        def dfs(u: str) -> None:
            state[u] = 1
            stack.append(u)
            for v in sorted(graph.get(u, ())):
                if state.get(v, 0) == 0:
                    dfs(v)
                elif state.get(v) == 1:
                    cyc = stack[stack.index(v):] + [v]
                    if cyc not in cycles:
                        cycles.append(cyc)
            stack.pop()
            state[u] = 2

        for m in sorted(graph):
            if state.get(m, 0) == 0:
                dfs(m)

        pretty = [" -> ".join(x.replace("libucks.", "") for x in c) for c in cycles]
        assert pretty == [], f"runtime import cycle(s): {pretty}"

    def test_no_upward_layer_imports(self):
        """A storage/model/parsing module must never import orchestration."""
        LAYER = {
            "libucks.models": 0, "libucks.config": 0, "libucks.eval_metrics": 0,
            "libucks.storage": 1, "libucks.embeddings": 1,
            "libucks.parsing": 1, "libucks.diff": 1,
            "libucks.cache_augmentation": 2, "libucks.thinking": 2,
        }

        def layer(m: str) -> int:
            for pre, lv in LAYER.items():
                if m == pre or m.startswith(pre + "."):
                    return lv
            return 3  # orchestration: librarian, mitosis, mcp_bridge, ...

        bad = [
            f"L{layer(src)} {src} -> L{layer(dst)} {dst}"
            for src, dsts in _runtime_graph().items()
            for dst in sorted(dsts)
            if layer(dst) > layer(src)
        ]
        assert bad == [], f"layering violation(s): {bad}"


# ---------------------------------------------------------------------------
# 2. Duplicated logic
# ---------------------------------------------------------------------------

# name -> why more than one definition is correct
ALLOWED_DUPLICATES = {
    "_read_chunk_content": (
        "two intentional families, geometry vs content — "
        "see test_read_chunk_content_families.py"
    ),
    "_collect_source_text": "librarian + data_generator; kept in sync, tested",
    "_log": "trivial per-module stderr helper",
    "_resolve_device": "model_manager + train_adapter; trivial device string",
    "_rough_tokens": "trivial len*0.25; four copies, all identical",
    "forward": "nn.Module subclasses",
    "main": "per-script entry points",
    "load": "unrelated classes",
    "save": "unrelated classes",
    "step": "unrelated classes",
    "train_step": "unrelated trainers",
    "param_count": "unrelated modules",
    "hidden_dim": "unrelated modules",
    "token_count_non_negative": "one validator per model class",
    "_build_qa_prompt": "duplicated inside _cli; see docs/cm-b-plan.md",
    "_parse_qa_pairs": "duplicated inside _cli; see docs/cm-b-plan.md",
    "_fetch_qa": "duplicated inside _cli; see docs/cm-b-plan.md",
    "_grounded": "thin delegating shims to eval_metrics.grounding_score",
    "_encode_centroid": "single def in bucket_registry + back-compat alias",
}


class TestNoNewDuplicateHelpers:
    def test_module_level_helpers_are_defined_once(self):
        """A second copy of a helper is how three of this repo's bugs started."""
        defs: dict[str, list[str]] = defaultdict(list)
        for p in _pkg_files():
            for node in _parse(p).body:  # module level only
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[node.name].append(f"{_modname(p)}:{node.lineno}")

        dupes = {
            name: locs for name, locs in sorted(defs.items())
            if len(locs) > 1 and name not in ALLOWED_DUPLICATES
        }
        assert dupes == {}, (
            "helper defined in more than one module. Import it instead, or add it "
            f"to ALLOWED_DUPLICATES with a reason: {dupes}"
        )


# ---------------------------------------------------------------------------
# 3. Cross-file invariants that nothing else enforces
# ---------------------------------------------------------------------------

class TestCrossFileInvariants:
    def test_merge_limit_stays_below_the_split_threshold(self):
        """Otherwise merge and mitosis fight forever; see
        test_merge_split_invariant.py for the full mechanism."""
        from libucks.merging_service import MERGE_TOKEN_RATIO
        assert 0.0 < MERGE_TOKEN_RATIO < 1.0

    def test_every_config_key_has_a_production_reader(self):
        """A knob nobody reads is worse than no knob — RUNBOOK recommended
        tuning `compression_steps` to fix OOM, and it does nothing."""
        INERT = {
            "compression_steps": "LatentCompressor is never constructed; documented in config.py",
            "grammar_cache": "PathsConfig; documented in ARCHITECTURE.md, unread",
            "log_file": "PathsConfig; documented in ARCHITECTURE.md, unread",
            "pending_events": "PathsConfig; documented in ARCHITECTURE.md, unread",
            "repo_cache": "PathsConfig; documented in ARCHITECTURE.md, unread",
        }
        cfg_path = PKG / "config.py"
        fields = [
            stmt.target.id
            for node in _parse(cfg_path).body
            if isinstance(node, ast.ClassDef)
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]

        other_src = "\n".join(
            p.read_text() for p in _pkg_files() if p != cfg_path
        )
        unread = [
            f for f in fields
            if f not in INERT and f".{f}" not in other_src and f'"{f}"' not in other_src
        ]
        assert unread == [], (
            "config key with no production reader — wire it up or document it "
            f"as inert: {unread}"
        )

    def test_deferred_imports_are_not_called_at_runtime(self):
        """TYPE_CHECKING-only name used for real = NameError in production."""
        offenders = []
        for p in _pkg_files():
            tree = _parse(p)
            tc: set[str] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.If):
                    t = n.test
                    if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
                    ):
                        for sub in ast.walk(n):
                            if isinstance(sub, ast.alias):
                                tc.add((sub.asname or sub.name).split(".")[-1])
            if not tc:
                continue
            # a real import anywhere earlier rebinds the name at runtime
            real: dict[str, list[int]] = defaultdict(list)
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        nm = (a.asname or a.name).split(".")[-1]
                        if nm in tc:
                            real[nm].append(n.lineno)
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in tc:
                    if any(ln < n.lineno for ln in real.get(n.func.id, [])):
                        continue
                    offenders.append(f"{_modname(p)}:{n.lineno} calls {n.func.id}()")
        assert offenders == [], (
            f"TYPE_CHECKING-only name called at runtime: {offenders}"
        )


# ---------------------------------------------------------------------------
# 4. Measurement integrity — the code that produces the numbers
# ---------------------------------------------------------------------------

class TestMeasurementIntegrity:
    """A second grounding scorer is how CM-A.2 was mis-scored by 3 fixtures."""

    def test_only_eval_metrics_implements_grounding(self):
        offenders = []
        for p in list(_pkg_files()) + sorted((ROOT / "scripts").rglob("*.py")) \
                 + sorted((ROOT / "tests").rglob("*.py")):
            if p.name == "eval_metrics.py" or "archive" in p.parts:
                continue
            try:
                tree = _parse(p)
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if n.name not in ("_grounded", "_grounding_score", "grounding_score",
                                  "keyword_hit", "keyword_variants"):
                    continue
                body = ast.get_source_segment(p.read_text(), n) or ""
                # a delegating shim is fine; a reimplementation is not
                if "grounding_score" in body or "keyword_hit" in body:
                    continue
                offenders.append(f"{p.relative_to(ROOT)}:{n.lineno} {n.name}")
        assert offenders == [], (
            "grounding metric reimplemented outside libucks/eval_metrics.py — "
            f"delegate to it instead: {offenders}"
        )

    @pytest.mark.parametrize("script", ["cm_distill_buckets.py", "cm_eval_cartridge.py",
                                        "run_eval.py"])
    def test_mps_watermark_is_set_before_torch_import(self, script: str):
        """PYTORCH_MPS_HIGH_WATERMARK_RATIO after `import torch` is a no-op,
        and has cost this project whole nights."""
        p = ROOT / "scripts" / script
        if not p.exists():
            pytest.skip(f"{script} not present")
        src = p.read_text()
        env_at = src.find("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        torch_at = min(
            (i for i in (src.find("\nimport torch"), src.find("\n    import torch"))
             if i != -1),
            default=-1,
        )
        assert env_at != -1, f"{script} never sets PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        if torch_at != -1:
            assert env_at < torch_at, (
                f"{script} sets the MPS watermark AFTER importing torch — no effect"
            )


class TestNoReciprocalConstants:
    """A constant restated in two files under "MUST match" comments is not an
    invariant, it is a promise. `PREFIX_LEN = 128` was declared in both
    cm_distill_buckets.py and cm_eval_cartridge.py, each pointing at the other;
    cartridge.py's own load() docstring called a mismatch "a question of when,
    not if". A P sweep would have made it when.

    The fix is to read the value from the artifact
    (`KVPrefixCartridge.read_geometry`), not to restate it. This test fails if a
    new reciprocal pair appears.
    """

    def test_no_module_claims_a_constant_must_match_another_module(self):
        offenders = []
        for p in sorted((ROOT / "scripts").rglob("*.py")) + list(_pkg_files()):
            if "archive" in p.parts:
                continue
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if "MUST match" not in line:
                    continue
                # Only flag it when the SAME line also assigns a literal — that
                # is the restated-constant shape. Prose references are fine.
                if "=" in line.split("#")[0] and any(c.isdigit() for c in line.split("#")[0]):
                    offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
        assert not offenders, (
            "constant restated under a MUST-match comment; read it from the "
            "artifact instead:\n  " + "\n  ".join(offenders)
        )

    def test_eval_reads_prefix_len_from_the_cartridge_file(self):
        src = (ROOT / "scripts" / "cm_eval_cartridge.py").read_text()
        assert "read_geometry" in src, (
            "cm_eval_cartridge must derive P from the cartridge it loads, so a "
            "cartridge trained at a different P is evaluable without a code edit"
        )


class TestDistillIsReproducible:
    """Nothing in the CM pipeline was seeded, so no result in this track has an
    error bar. `distill_bucket` must at least ACCEPT a seed, and the batch
    script must expose it."""

    def test_distill_bucket_accepts_a_seed(self):
        import inspect

        from libucks.thinking.training.cartridge_trainer import CartridgeTrainer
        assert "seed" in inspect.signature(CartridgeTrainer.distill_bucket).parameters

    def test_batch_distiller_exposes_and_forwards_the_seed(self):
        src = (ROOT / "scripts" / "cm_distill_buckets.py").read_text()
        assert "CM_SEED" in src, "no way to request a reproducible run"
        assert "seed=SEED" in src, "the seed is read but never forwarded to the trainer"

    def test_recipe_banner_reports_the_seed(self):
        """CM-A.2's log claimed a configuration it did not run; the banner exists
        so that can't recur, and an unseeded run must say so out loud."""
        src = (ROOT / "scripts" / "cm_distill_buckets.py").read_text()
        assert "UNSEEDED" in src
