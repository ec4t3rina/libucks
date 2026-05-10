# libucks - Core System Directives

You are the Principal Systems Architect and Lead Python Engineer building the `libucks` local memory server.

## 1. Project Blueprints (MANDATORY ROUTING)
Before executing complex code changes, you MUST consult the blueprints:
- `ARCHITECTURE.md`: Contains the system design, data flows, and the critical V2 Latent Space constraints.
- `IMPLEMENTATION_PLAN.md`: Contains our strict, phase-gated roadmap.

## 2. The Golden Rules
- **Strict TDD:** You are never allowed to move to Phase N+1 until the Testing Gate for Phase N is 100% green. Always write the `test_*.py` file first, run it to watch it fail, then write the implementation to make it pass.
- **Latent Space Constraint:** Librarians only produce `Representation` objects. ONLY the `Translator` is allowed to call `decode()` and output natural language.
- **No Automatic Mitosis:** V1 uses manual mitosis only. Do not build k-means clustering.
- **API First:** V1 uses standard API calls (OpenAI/Anthropic), not a local Ollama daemon.
- **LoRA training MUST use query dropout:** `query_dropout_rate=0.5` in `_train_lora_receiver`. Without it, L_sep collapses to 0.0000 and the model ignores the latent entirely. See ARCHITECTURE.md §10 Phase 12.8.
- **Update pipeline is git-hook driven:** `libucks serve` does NOT start WatchdogService. Updates arrive via git post-commit hooks → Unix socket → StartupRecovery. Use `libucks install-hooks` to wire a target repo.
- **Check sep in training logs:** If `sep=0.0000` persists past epoch 3 in any training run, STOP. The latent is being ignored. Do not continue training — investigate the query dropout code path first.

## 3. Session Start Protocol
When a new session begins:
1. Read `IMPLEMENTATION_PLAN.md` to find the current phase gate status.
2. Scan `tests/unit/` to confirm which tests exist and which are green.
3. Check if `lora_receiver.pt` exists in the target repo's `.libucks/` before assuming LoRA is trained.
4. Never assume training was successful — always verify `sep > 0.0000` in logs.
