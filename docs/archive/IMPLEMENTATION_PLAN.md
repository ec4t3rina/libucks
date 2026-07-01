# libucks — Implementation Plan

## Current Status (as of 2026-04-17)

**Completed:**
- All data models, storage, embeddings, routing (Phases 1–2)
- Full V1 `TextStrategy` + Anthropic API integration
- Full V2.1 `LatentStrategy`: encode/reason/decode, LoRA receiver, CommunicationAdapter
- CentralAgent (routing, event dispatch, mitosis guard, retry buffer)
- Librarian (UpdateEvent, TombstoneEvent, PathUpdateEvent, QueryEvent)
- WatchdogService, DiffExtractor, StartupRecovery, StaleChecker
- MCP Bridge (`libucks_query`, `libucks_status`)
- HealthMonitor, MitosisService, MergingService
- Git hook receiver + socket listener
- `libucks init`, `libucks query`, `libucks serve`, `libucks train-adapter`
- Training pipeline: adapter (Phase 1) + LoRA receiver (Phase 2) with Q&A teacher

**Active training issue (Phase 12.8):**
L_sep collapsed to 0.0000 throughout training because tight Q&A pairs allowed the model
to reconstruct answers from query tokens alone, making the latent irrelevant.
Fix: query dropout (50%) + hard-negative selection + λ_sep=0.3. See §10 of ARCHITECTURE.md.

---

## Phase A: Fix LoRA Training (NOW)

### A.1 Retrain with query dropout

```bash
rm -f src/click/.libucks/lora_receiver.pt
libucks train-adapter --repo src/click --receiver-only --epochs 15
```

**Success criterion:** `sep` rises above 0.0000 within the first 3 epochs.
If it stays at 0.0000, stop and investigate — the latent is still being ignored.

**Expected loss trajectory:**
- L_task: should converge to 0.3–0.8 (higher than before — dropout steps are harder)
- L_sep: should rise to 0.001–0.05 range by epoch 5

### A.2 Validate inference

Test the three canonical queries. Success = factually grounded answers about Click's
actual implementation, not Base LM hallucinations.

```bash
libucks query "Why does Click use ctypes to call Windows APIs directly?" --repo src/click --top-k 1
libucks query "What is the difference between BEFORE_BAR and AFTER_BAR on Windows versus other systems?" --repo src/click --top-k 3
libucks query "When does the File class actually open a file, and why does it defer this action?" --repo src/click --top-k 1
```

---

## Phase B: UPDATE Workflow Test Coverage (TDD)

The UPDATE pipeline classes are all implemented. Tests are missing for two components.

### B.1 `test_diff_extractor.py` (write first, then verify impl passes)

File: `tests/unit/test_diff_extractor.py`

**Cases to cover:**

| Test | Input | Expected |
|---|---|---|
| Added lines only | unified diff with `+` lines | DiffHunk.added_lines populated, removed empty |
| Removed lines only | unified diff with `-` lines | DiffHunk.removed_lines populated, added empty |
| Mixed hunk | diff with both | both populated correctly |
| Rename detection | diff with `rename from/to` headers | `DiffEvent.is_rename=True`, old_path/new_path set |
| Binary file | "Binary files differ" | returns `[]` silently |
| Empty diff | empty string | returns `[]` |
| Multi-hunk file | two hunks in one diff | two DiffHunk objects in one DiffEvent |

Test strategy: mock `git.Repo` to return fixed diff strings; call `_parse_diff_output` directly.
No real git repo needed for unit tests.

### B.2 `test_watchdog_service.py` (write first)

File: `tests/unit/test_watchdog_service.py`

**Cases to cover:**

| Test | Scenario | Expected |
|---|---|---|
| Extension filter | `.DS_Store` modified | no event posted |
| Extension filter | `.py` modified | event extraction attempted |
| Debounce | same file modified twice within 500ms | DiffExtractor called once |
| Directory event | directory modify | ignored |
| Extract error | extractor raises | logged, no crash |

Test strategy: inject a mock `DiffExtractor` and mock `CentralAgent`. Trigger
`_Handler.on_modified()` directly.

### B.3 Integration test: git commit → bucket updated

File: `tests/integration/test_update_pipeline.py`

Setup: initialise a real git repo in `tmp_path` with one Python file.
Run `libucks init`. Record the initial bucket prose.
Make a change to the file, `git commit`.
Trigger `StartupRecovery.run()` directly (same path as the serve command uses).
Assert the relevant bucket's prose is updated to reflect the new code.

This is the end-to-end UPDATE test. It uses real git, real embeddings (or injected mock),
and real Librarian write path.

---

## Phase C: MCP Bridge Hardening

### C.1 Adapter dimension in `serve()` (bug)

In `mcp_bridge.py:147`:
```python
adapter = CommunicationAdapter()   # uses default hidden_dim=2048
```

This ignores the actual encoder/base model dimensions from config. If the deployed models
have different hidden sizes, the adapter will load weights with wrong dimensions and silently
produce garbage. Fix: read dims from `AutoConfig` same as `_run_train_adapter` does.

```python
from transformers import AutoConfig as _AC
_enc_dim  = _AC.from_pretrained(cfg.model.local_model).hidden_size
_base_dim = _AC.from_pretrained(cfg.model.base_model).hidden_size
adapter = CommunicationAdapter(hidden_dim=_enc_dim, output_dim=_base_dim)
```

### C.2 Watchdog not started in `serve()`

`WatchdogService` is implemented but never started inside `serve()`. The current update
path relies entirely on git hooks → socket → `StartupRecovery.run()`. This is intentional
(updates at commit boundaries, not on every save), but must be documented clearly.

Decision: keep git-hook–driven updates as the primary path. Document it in ARCHITECTURE.md §3.2.
Add a note that WatchdogService is available as an opt-in for repos that do not use git hooks
(e.g. non-git directories), but is not started by default.

### C.3 `libucks serve` status check

`libucks serve` is a stdio subprocess (no port to poll). To check if it is active:

```bash
# Is the process running?
ps aux | grep "libucks serve"

# Is the socket alive?  (written by mcp_bridge.py at startup)
ls -la .libucks/server.sock

# What HEAD was last indexed?
python3 -c "import json; d=json.load(open('.libucks/registry.json')); print(d.get('_meta', {}))"
```

The `registry.json` `_meta` block stores `watcher_pid` and `last_indexed_head`.
For day-to-day local testing, use `libucks query` directly — it bypasses the MCP server
entirely and has no 60-second timeout.

---

## Phase D: Final End-to-End Validation

### D.1 MCP tool smoke test

Start `libucks serve` with `LIBUCKS_REPO_PATH` pointing at `src/click`.
Use `claude mcp` or the MCP inspector to call `libucks_query` and `libucks_status`.
Verify the wire format is correct and the answer is grounded.

### D.2 Git hook integration

```bash
libucks install-hooks --repo src/click
cd src/click
# make a real change
echo "# libucks_test" >> click/core.py
git add click/core.py && git commit -m "test: libucks hook integration"
# verify the bucket updated
libucks query "What is in click/core.py?" --repo src/click --top-k 1
```

### D.3 Claude Code MCP wiring

Add to `.claude/claude_desktop_config.json` (or Claude Code MCP settings):
```json
{
  "mcpServers": {
    "libucks": {
      "command": "/path/to/.venv/bin/libucks",
      "args": ["serve"],
      "env": { "LIBUCKS_REPO_PATH": "/path/to/target/repo" }
    }
  }
}
```

Verify `libucks_query` appears in Claude Code's tool list and returns a grounded answer.

---

## Testing Gate Summary

| Phase | Gate | Status |
|---|---|---|
| A | L_sep > 0.000 in training logs; Q1–Q3 answers are factually grounded | ✅ L_align fix working |
| B.1 | `pytest tests/unit/test_diff_extractor.py` 100% green | ✅ 17 tests |
| B.2 | `pytest tests/unit/test_watchdog_service.py` 100% green | ✅ 13 tests |
| B.3 | `pytest tests/integration/test_update_pipeline.py` 100% green | ✅ 5 tests |
| C.1 | `CommunicationAdapter` dimensions loaded from config in serve() | ✅ fixed |
| C.2 | WatchdogService decision documented | ✅ documented in mcp_bridge.py |
| D | MCP tool returns grounded answers from Claude Code | 🔴 not tested |
