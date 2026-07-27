"""CM-B Stage 1, step 1 — generate controlled single-chunk edits.

The Edit experiment asks whether a stale cartridge can be REPAIRED more cheaply
than it can be rebuilt (~7,200 s). To answer that with a curve rather than an
anecdote, we need edits whose magnitude we choose. Real history cannot supply
them: libugry has one commit and echoswarm's fourteen are almost entirely
"Update README.md".

Four types, ordered by how much they should disturb the cartridge:

    rename    identifier changes, behaviour does not      — smallest
    constant  a numeric fact changes                      — small, but a FACT,
                                                            which is what the
                                                            eval fixtures probe
    branch    new control flow appears                    — medium
    delete    a helper other code calls disappears        — largest

Design notes, all of which the tests enforce:

  * Deterministic given a seed. A re-run must reproduce the numbers, so target
    selection is seeded and stable, never "pick the first match we happen to
    find" in dict order.
  * Reversible byte-for-byte. Ground truth and every repair method must start
    from an identical tree, so `revert_edit` restores exactly.
  * Never a no-op. An edit that does not change the file produces a "repair was
    free" datapoint that means nothing — the staleness-floor trap from
    docs/cm-b-plan.md, caught one step earlier.
  * Output stays valid Python. A syntax error would break _read_chunk_content
    and quietly poison the re-distill.

This module is pure Python and imports nothing from torch, so it runs while a
distillation job holds the GPU.

CLI:
    uv run python scripts/cm_make_edits.py --repo <path> --file relay.py \
        --type constant [--seed 0] [--commit] [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

EDIT_TYPES: tuple[str, ...] = ("rename", "constant", "branch", "delete")

# Upper bound on a deleted function, in lines. `delete` is meant to be the
# LARGEST of the four disturbances, but a 65-line removal (echoswarm's
# api.py:refresh_map) spans several chunks, and the whole claim under test is
# that a SINGLE changed chunk can be repaired cheaply. Beyond this the
# datapoint stops measuring what Stage 1 is about.
MAX_DELETE_LINES: int = 25


def _log(msg: str) -> None:
    print(f"[cm_make_edits] {msg}", file=sys.stderr, flush=True)


@dataclass
class EditPlan:
    """A single, reversible, one-chunk edit.

    old_text/new_text are exact substrings of the file, so applying is a single
    `str.replace(old, new, 1)` and reverting is the same call inverted. Storing
    text rather than line numbers keeps revert correct even though applying can
    change the line count (branch adds lines, delete removes them).
    """

    path: Path
    edit_type: str
    old_text: str
    new_text: str
    line_start: int
    line_end: int
    description: str

    @property
    def line_delta(self) -> int:
        return len(self.new_text.splitlines()) - len(self.old_text.splitlines())


# ---------------------------------------------------------------------------
# Target discovery
# ---------------------------------------------------------------------------

def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every def in the file — module-level AND methods.

    Real code is mostly methods. echoswarm's simulation.py has four
    module-level functions and twelve methods, and the methods are where
    _relay_messages and _move_agents live — the exact identifiers the eval
    fixtures probe. Restricting to `tree.body` made rename and delete
    unusable on the files this experiment actually needs.
    """
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _sole_method_names(tree: ast.Module) -> set[str]:
    """Methods that are their class's only body member.

    Deleting one leaves an empty class body, which is a SyntaxError rather
    than an edit.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        defs = [b for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
        others = [b for b in node.body if b not in defs
                  and not (isinstance(b, ast.Expr)
                           and isinstance(b.value, ast.Constant))]
        if len(defs) == 1 and not others:
            out.add(defs[0].name)
    return out


def _pick(items: list, seed: int):
    """Seeded, stable choice. Sorting first makes it independent of AST order."""
    if not items:
        return None
    return random.Random(seed).choice(items)


def _plan_rename(path: Path, src: str, tree: ast.Module, seed: int) -> Optional[EditPlan]:
    """Rename a function that is called at least once elsewhere in the file.

    Requiring a call site is the point: a rename nothing references is a
    comment-level change and would understate the disturbance.
    """
    candidates = sorted(
        (fn for fn in _functions(tree) if src.count(fn.name) >= 2),
        key=lambda fn: fn.name,
    )
    fn = _pick(candidates, seed)
    if fn is None:
        return None

    verb, _, rest = fn.name.partition("_")
    new_name = f"scan_{rest}" if rest and verb != "scan" else f"{fn.name}_v2"

    return EditPlan(
        path=path, edit_type="rename",
        old_text=fn.name, new_text=new_name,
        line_start=fn.lineno, line_end=fn.end_lineno or fn.lineno,
        description=f"rename {fn.name} -> {new_name} ({src.count(fn.name)} occurrences)",
    )


_NUM_ASSIGN = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<val>\d+(?:\.\d+)?)\s*$")


def _plan_constant(path: Path, src: str, tree: ast.Module, seed: int) -> Optional[EditPlan]:
    """Change a module-level numeric constant — a FACT the fixtures can probe."""
    hits = []
    for i, line in enumerate(src.splitlines(), start=1):
        m = _NUM_ASSIGN.match(line.strip())
        if m:
            hits.append((i, line, m))
    hits.sort(key=lambda t: t[2].group("name"))
    chosen = _pick(hits, seed)
    if chosen is None:
        return None

    lineno, line, m = chosen
    old_val = m.group("val")
    if "." in old_val:
        new_val = f"{max(0.1, round(float(old_val) - 0.2, 2))}"
    else:
        new_val = str(int(old_val) * 2 + 1)

    return EditPlan(
        path=path, edit_type="constant",
        old_text=line, new_text=line.replace(old_val, new_val, 1),
        line_start=lineno, line_end=lineno,
        description=f"constant {m.group('name')}: {old_val} -> {new_val}",
    )


def _plan_branch(path: Path, src: str, tree: ast.Module, seed: int) -> Optional[EditPlan]:
    """Insert a guard clause at the top of a function body."""
    lines = src.splitlines(keepends=True)
    # Guarding on `self`/`cls` is nonsense, so a method needs a real second
    # parameter to be a candidate.
    def _first_real_arg(fn: ast.FunctionDef) -> Optional[str]:
        names = [a.arg for a in fn.args.args]
        if names and names[0] in ("self", "cls"):
            names = names[1:]
        return names[0] if names else None

    candidates = sorted(
        (fn for fn in _functions(tree) if _first_real_arg(fn)),
        key=lambda fn: fn.name,
    )
    fn = _pick(candidates, seed)
    if fn is None:
        return None

    body0 = fn.body[0]
    # Skip a docstring so the guard lands in real code.
    if isinstance(body0, ast.Expr) and isinstance(body0.value, ast.Constant) \
       and isinstance(body0.value.value, str) and len(fn.body) > 1:
        body0 = fn.body[1]

    anchor = lines[body0.lineno - 1]
    indent = anchor[: len(anchor) - len(anchor.lstrip())]
    arg = _first_real_arg(fn)
    guard = f"{indent}if {arg} is None:\n{indent}    return None\n"

    return EditPlan(
        path=path, edit_type="branch",
        old_text=anchor, new_text=guard + anchor,
        line_start=body0.lineno, line_end=body0.lineno,
        description=f"branch: guard clause on {arg} in {fn.name}",
    )


def _plan_delete(path: Path, src: str, tree: ast.Module, seed: int) -> Optional[EditPlan]:
    """Delete a function that something else calls — the largest disturbance.

    The result is intentionally still parseable but semantically broken (a
    dangling call). That mirrors a real mid-refactor commit, and the cartridge
    should notice the definition is gone.
    """
    sole = _sole_method_names(tree)

    def _span(fn: ast.FunctionDef) -> int:
        start = min([d.lineno for d in fn.decorator_list] + [fn.lineno])
        return (fn.end_lineno or fn.lineno) - start + 1

    candidates = sorted(
        (fn for fn in _functions(tree)
         if src.count(fn.name) >= 2
         and fn.name not in sole
         and _span(fn) <= MAX_DELETE_LINES),
        key=lambda fn: fn.name,
    )
    fn = _pick(candidates, seed)
    if fn is None:
        return None

    lines = src.splitlines(keepends=True)
    start = fn.lineno - 1
    if fn.decorator_list:
        start = min(d.lineno for d in fn.decorator_list) - 1
    end = fn.end_lineno or fn.lineno
    block = "".join(lines[start:end])

    return EditPlan(
        path=path, edit_type="delete",
        old_text=block, new_text="",
        line_start=start + 1, line_end=end,
        description=f"delete {fn.name} ({end - start} lines)",
    )


_PLANNERS = {
    "rename": _plan_rename,
    "constant": _plan_constant,
    "branch": _plan_branch,
    "delete": _plan_delete,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_edit(path: Path, edit_type: str, seed: int = 0) -> EditPlan:
    """Choose a concrete edit without touching the file.

    Raises ValueError if the type is unknown or the file offers no suitable
    target — better a loud failure than a silent no-op datapoint.
    """
    path = Path(path)
    if edit_type not in _PLANNERS:
        raise ValueError(
            f"unknown edit type {edit_type!r}; expected one of {list(EDIT_TYPES)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"edit target does not exist: {path}")

    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise ValueError(f"{path} is not parseable Python: {exc}") from exc

    plan = _PLANNERS[edit_type](path, src, tree, seed)
    if plan is None:
        raise ValueError(
            f"no {edit_type!r} target found in {path.name}. Pick another file: "
            "an edit that changes nothing produces a meaningless datapoint."
        )
    if plan.old_text == plan.new_text:
        raise ValueError(f"{edit_type!r} planned a no-op on {path.name}")
    return plan


def apply_edit(plan: EditPlan) -> None:
    """Apply the plan in place. Verifies the result still parses."""
    src = plan.path.read_text()
    if plan.old_text not in src:
        raise ValueError(
            f"cannot apply {plan.edit_type}: anchor text absent from "
            f"{plan.path.name} (already applied, or the file changed underneath)"
        )
    # rename must hit every occurrence; the others are single-site.
    updated = (
        src.replace(plan.old_text, plan.new_text)
        if plan.edit_type == "rename"
        else src.replace(plan.old_text, plan.new_text, 1)
    )
    if updated == src:
        raise ValueError(f"{plan.edit_type} produced no change — refusing a no-op")

    try:
        ast.parse(updated)
    except SyntaxError as exc:
        raise ValueError(
            f"{plan.edit_type} would leave {plan.path.name} unparseable: {exc}"
        ) from exc

    plan.path.write_text(updated)
    _log(f"applied {plan.description}")


def revert_edit(plan: EditPlan) -> None:
    """Undo the plan, restoring the file byte-for-byte."""
    src = plan.path.read_text()
    if plan.edit_type == "rename":
        restored = src.replace(plan.new_text, plan.old_text)
    elif plan.new_text == "":
        # A deletion cannot be located by searching for "" — re-insert at the
        # recorded line instead.
        lines = src.splitlines(keepends=True)
        idx = plan.line_start - 1
        restored = "".join(lines[:idx]) + plan.old_text + "".join(lines[idx:])
    else:
        restored = src.replace(plan.new_text, plan.old_text, 1)

    plan.path.write_text(restored)
    _log(f"reverted {plan.edit_type} in {plan.path.name}")


def commit_edit(plan: EditPlan, repo: Path) -> str:
    """Stage and commit the applied edit. Returns the new HEAD sha.

    The commit is what makes the edit visible to libucks: BucketKVCache keys
    freshness on (chunk_id, git_sha), so an uncommitted edit leaves the
    cartridge looking valid.
    """
    repo = Path(repo)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m",
         f"cm-b edit [{plan.edit_type}]: {plan.description}"],
        check=True, capture_output=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    _log(f"committed {plan.edit_type} -> {sha[:8]}")
    return sha


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--file", required=True,
                    help="path to edit, relative to --repo")
    ap.add_argument("--type", required=True, choices=EDIT_TYPES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--commit", action="store_true",
                    help="commit the edit (required for libucks to see it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without touching the file")
    args = ap.parse_args()

    target = args.repo / args.file
    try:
        plan = plan_edit(target, args.type, seed=args.seed)
    except (ValueError, FileNotFoundError) as exc:
        _log(f"ERROR: {exc}")
        return 1

    _log(f"plan: {plan.description}")
    _log(f"  lines {plan.line_start}-{plan.line_end}, line delta {plan.line_delta:+d}")
    if args.dry_run:
        _log("dry run — nothing written")
        return 0

    apply_edit(plan)
    if args.commit:
        commit_edit(plan, args.repo)
    else:
        _log("NOT committed — libucks keys cartridge freshness on git_sha, so "
             "an uncommitted edit leaves the cartridge looking fresh. Use --commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
