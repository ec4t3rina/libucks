# Archived scripts

Superseded tooling, kept for history rather than use. Nothing here is on a current
code path — check `git log` for the context each one belongs to.

| script | archived | why |
|---|---|---|
| `query_interlat.py` | 2026-07-27 | V2 Interlat-era manual query driver. Zero references anywhere in the repo or docs. Loads `Qwen2.5-3B-Instruct` with `quantization="4bit"`, which needs the `mps-bitsandbytes` fork un-vendored in the CM-B cleanup, so it will not run as written. The current entry points are `libucks query` (`libucks/_cli.py`) and `scripts/run_eval.py`. |

Still active and deliberately NOT archived:
- `check_nervous_system.py` — V1-era, but it exercises the git-hook + Unix-socket
  update pipeline, which is still current architecture.
- `diagnose_adapter.py` — the documented first step for the recurring "deaf adapter"
  failure (cross-bucket cosine ~1.0).
- `train_lora_receiver.py` — referenced by `ARCHITECTURE.md:718`; the LoRA receiver is
  still part of the hybrid path.
