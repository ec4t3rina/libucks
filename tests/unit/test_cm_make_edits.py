"""Stage 1, step 1: controlled single-chunk edits for the repair experiment.

The Edit experiment needs code changes whose *magnitude* we control, so the
result can be a curve — repair cost vs how much the edit disturbed the
cartridge — rather than a single anecdote. Real history cannot supply that:
libugry has 1 commit and echoswarm's 14 are nearly all "Update README.md".

Four edit types, ordered by expected disturbance:

    rename    — identifier changes, semantics do not
    constant  — a numeric fact changes (this is what fixtures actually probe)
    branch    — new control flow appears
    delete    — a helper other code calls disappears

The invariants that matter for the experiment, and that these tests pin:

  * an edit must actually change the file (a no-op edit silently produces a
    "repair is free!" datapoint that means nothing — this is the staleness
    floor problem in the plan, moved one step earlier)
  * it must be confined to ONE chunk, or "single-chunk repair" is untested
  * it must be reversible, so the ground-truth re-distill and every repair
    method start from an identical tree
  * it must be deterministic given a seed, so a re-run reproduces the numbers
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.cm_make_edits import (
    EDIT_TYPES,
    EditPlan,
    apply_edit,
    plan_edit,
    revert_edit,
)


SAMPLE = '''\
"""Agent relay logic."""

RELAY_PROBABILITY = 0.8
MAX_AGENTS = 20


def check_vulnerabilities(agent):
    """Return True when the agent is compliant."""
    if agent.score > 10:
        return True
    return False


def relay_message(agent, message):
    if check_vulnerabilities(agent):
        return f"{agent.name}: {message}"
    return None


def summarise(agents):
    return [relay_message(a, "ping") for a in agents]
'''


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo — edits must be committable and revertible."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "relay.py").write_text(SAMPLE)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.fixture
def target(repo: Path) -> Path:
    return repo / "relay.py"


ALL_TYPES = pytest.mark.parametrize("edit_type", EDIT_TYPES)


class TestPlanning:
    @ALL_TYPES
    def test_plan_finds_a_target_for_every_edit_type(self, target: Path, edit_type: str):
        plan = plan_edit(target, edit_type, seed=0)
        assert isinstance(plan, EditPlan)
        assert plan.edit_type == edit_type
        assert plan.old_text, "must anchor on real text"
        # `delete` legitimately has empty new_text — that IS the edit.
        assert plan.old_text != plan.new_text
        if edit_type != "delete":
            assert plan.new_text

    @ALL_TYPES
    def test_planning_is_deterministic(self, target: Path, edit_type: str):
        """Same seed must reproduce the run; otherwise numbers are not comparable."""
        a = plan_edit(target, edit_type, seed=7)
        b = plan_edit(target, edit_type, seed=7)
        assert (a.old_text, a.new_text, a.line_start) == (b.old_text, b.new_text, b.line_start)

    @ALL_TYPES
    def test_plan_reports_the_lines_it_will_touch(self, target: Path, edit_type: str):
        plan = plan_edit(target, edit_type, seed=0)
        n_lines = len(target.read_text().splitlines())
        assert 1 <= plan.line_start <= plan.line_end <= n_lines

    def test_unknown_edit_type_is_rejected(self, target: Path):
        with pytest.raises(ValueError, match="unknown edit type"):
            plan_edit(target, "explode", seed=0)

    def test_missing_target_is_rejected(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            plan_edit(tmp_path / "nope.py", "rename", seed=0)


class TestApplying:
    @ALL_TYPES
    def test_edit_actually_changes_the_file(self, target: Path, edit_type: str):
        """A no-op edit yields a meaningless 'repair was free' datapoint."""
        before = target.read_text()
        apply_edit(plan_edit(target, edit_type, seed=0))
        assert target.read_text() != before

    @ALL_TYPES
    def test_result_is_still_valid_python(self, target: Path, edit_type: str):
        """A syntax error would break _read_chunk_content and the re-distill."""
        import ast
        apply_edit(plan_edit(target, edit_type, seed=0))
        ast.parse(target.read_text())

    @ALL_TYPES
    def test_edit_is_confined_to_a_small_span(self, target: Path, edit_type: str):
        """'Single-chunk repair' is the claim; a sprawling edit does not test it.

        Uses a real diff, not positional comparison — inserting two lines shifts
        every line below it and would look like a whole-file rewrite.
        """
        import difflib

        before = target.read_text().splitlines()
        apply_edit(plan_edit(target, edit_type, seed=0))
        after = target.read_text().splitlines()

        changed = sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, before, after
            ).get_opcodes()
            if tag != "equal"
        )
        assert changed <= 8, f"{edit_type} touched {changed} lines"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_rename_leaves_no_trace_of_the_old_name(self, target: Path, seed: int):
        """A half-rename leaves the file broken, not merely edited.

        Asserted as a property across seeds rather than against one hardcoded
        function, so the test does not silently depend on which target the
        seeded chooser happens to pick.
        """
        plan = plan_edit(target, "rename", seed=seed)
        old, new = plan.old_text, plan.new_text
        before = target.read_text()
        assert before.count(old) >= 2, "rename must target a name with a call site"

        apply_edit(plan)
        src = target.read_text()
        assert old not in src, f"{old} survived the rename"
        assert src.count(new) >= 2, "definition and call site must both be renamed"

    def test_constant_changes_a_number(self, target: Path):
        plan = plan_edit(target, "constant", seed=0)
        apply_edit(plan)
        assert plan.old_text not in target.read_text()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_delete_removes_a_called_helper(self, target: Path, seed: int):
        """The deleted function must have had a caller — that is the point."""
        plan = plan_edit(target, "delete", seed=seed)
        name = plan.description.split()[1]
        assert target.read_text().count(name) >= 2, "must delete a CALLED helper"

        apply_edit(plan)
        assert f"def {name}" not in target.read_text()


CLASS_SAMPLE = '''\
"""Simulation core."""

TICK_LIMIT = 50


class Simulation:
    def __init__(self, agents):
        self.agents = agents

    def tick(self):
        self._relay_messages()
        self._move_agents()
        return len(self.agents)

    def _relay_messages(self):
        for a in self.agents:
            a.relayed = True

    def _move_agents(self):
        for a in self.agents:
            a.x += 1


class Tiny:
    def only_method(self):
        return 1
'''


@pytest.fixture
def class_target(repo: Path) -> Path:
    """Real code is methods on classes, not module-level functions.

    echoswarm's simulation.py has four module-level functions and twelve
    methods; the methods are where _relay_messages and _move_agents live —
    the very identifiers the eval fixtures probe.
    """
    p = repo / "sim.py"
    p.write_text(CLASS_SAMPLE)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "sim"], cwd=repo, check=True)
    return p


class TestMethodsAreValidTargets:
    @ALL_TYPES
    def test_every_edit_type_works_on_a_class_based_file(
        self, class_target: Path, edit_type: str
    ):
        plan = plan_edit(class_target, edit_type, seed=0)
        apply_edit(plan)
        import ast
        ast.parse(class_target.read_text())

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_rename_can_select_a_method(self, class_target: Path, seed: int):
        """At least one seed must reach a method, or methods are unreachable."""
        names = {plan_edit(class_target, "rename", seed=s).old_text for s in range(6)}
        assert names & {"_relay_messages", "_move_agents", "tick"}, (
            f"planner never selects a method; only found {names}"
        )

    def test_branch_never_guards_on_self_or_cls(self, class_target: Path):
        """`if self is None: return None` is nonsense inside a method."""
        for seed in range(8):
            plan = plan_edit(class_target, "branch", seed=seed)
            assert "if self is None" not in plan.new_text
            assert "if cls is None" not in plan.new_text

    def test_delete_prefers_a_single_chunk_sized_function(self, repo: Path):
        """A 65-line deletion spans several chunks and stops testing the claim.

        echoswarm's api.py has a 65-line `refresh_map`; the planner must not
        pick it when smaller called helpers exist, or "single-chunk repair"
        is no longer what is being measured.
        """
        from scripts.cm_make_edits import MAX_DELETE_LINES

        big = "\n".join(f"    x{i} = {i}" for i in range(80))
        p = repo / "mixed.py"
        p.write_text(
            "def small_helper(a):\n    return a + 1\n\n\n"
            f"def huge(a):\n{big}\n    return small_helper(a)\n\n\n"
            "def caller(a):\n    return huge(a) + small_helper(a)\n"
        )
        for seed in range(8):
            plan = plan_edit(p, "delete", seed=seed)
            n = len(plan.old_text.splitlines())
            assert n <= MAX_DELETE_LINES, (
                f"seed {seed} chose a {n}-line deletion (cap {MAX_DELETE_LINES})"
            )

    def test_delete_never_empties_a_class_body(self, class_target: Path):
        """Removing a class's only method is a SyntaxError, not an edit."""
        for seed in range(8):
            plan = plan_edit(class_target, "delete", seed=seed)
            assert "only_method" not in plan.description, (
                "planner chose the sole method of class Tiny"
            )


class TestReverting:
    @ALL_TYPES
    def test_revert_restores_the_file_byte_for_byte(self, target: Path, edit_type: str):
        """Every repair method must start from an identical tree."""
        original = target.read_text()
        plan = plan_edit(target, edit_type, seed=0)
        apply_edit(plan)
        assert target.read_text() != original
        revert_edit(plan)
        assert target.read_text() == original

    @ALL_TYPES
    def test_apply_revert_apply_is_stable(self, target: Path, edit_type: str):
        plan = plan_edit(target, edit_type, seed=0)
        apply_edit(plan)
        first = target.read_text()
        revert_edit(plan)
        apply_edit(plan)
        assert target.read_text() == first


class TestCommitting:
    def test_commit_produces_exactly_one_new_commit(self, repo: Path, target: Path):
        from scripts.cm_make_edits import commit_edit

        def count() -> int:
            out = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                                 cwd=repo, capture_output=True, text=True, check=True)
            return int(out.stdout.strip())

        before = count()
        plan = plan_edit(target, "constant", seed=0)
        apply_edit(plan)
        sha = commit_edit(plan, repo)
        assert count() == before + 1
        assert len(sha) >= 7

    def test_commit_message_records_the_edit_type(self, repo: Path, target: Path):
        from scripts.cm_make_edits import commit_edit

        plan = plan_edit(target, "branch", seed=0)
        apply_edit(plan)
        commit_edit(plan, repo)
        msg = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout
        assert "branch" in msg

    def test_git_sha_changes_so_the_kv_cache_notices(self, repo: Path, target: Path):
        """BucketKVCache invalidates on (chunk_id, git_sha); the edit must move it."""
        from scripts.cm_make_edits import commit_edit

        head = lambda: subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                      capture_output=True, text=True, check=True).stdout.strip()
        before = head()
        plan = plan_edit(target, "constant", seed=0)
        apply_edit(plan)
        commit_edit(plan, repo)
        assert head() != before
