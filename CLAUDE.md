# libucks - Core System Directives

You are the Principal Systems Architect and Lead Python Engineer building the `libucks` local memory server.

## 1. Project Blueprints (MANDATORY ROUTING)
Before executing complex code changes, you MUST consult the blueprints:
- `ARCHITECTURE.md`: Contains the system design, data flows, and the critical V2 Latent Space constraints.
- `docs/cartridges-plan.md`: The current active roadmap (Cartridge Memory track). Superseded V1/V2/Phase-4 plans are archived under `docs/archive/` (including the closed Phase 4-C negative-result log at `docs/archive/phase-4c/`).

## 2. The Golden Rules
- **Strict TDD:** You are never allowed to move to Phase N+1 until the Testing Gate for Phase N is 100% green. Always write the `test_*.py` file first, run it to watch it fail, then write the implementation to make it pass.
- **Latent Space Constraint:** Librarians only produce `Representation` objects. ONLY the `Translator` is allowed to call `decode()` and output natural language.
- **Auto-mitosis and auto-merge are on.** `HealthMonitor` runs every 5 min (started by `libucks serve` at `mcp_bridge.py:257`). It calls `MitosisService.split()` (k-means k=2 in `mitosis.py`) on buckets that exceed `mitosis_threshold` tokens or fall below the coherence threshold, and `MergingService.run_merge_pass()` for cosine-similar pairs. This was originally scoped as "manual only" in V1; that constraint was relaxed when HealthMonitor landed (Phase 6-E/F).
- **Auto-bucket-creation is on.** `NovelBucketService` (started by `libucks serve` alongside `HealthMonitor`) drains `CentralAgent.create_bucket_queue` to spawn new buckets when commits land substantial + cosine-novel content. Gates: `min_bucket_seed_tokens` (default 1500) AND `is_novel(embedding)`. Small or similar diffs are routed into the nearest existing bucket — coherence-driven mitosis splits later if pressure builds. Buckets are subject-specific knowledge units; single-file/single-chunk spawning is forbidden.
- **API First:** V1 uses standard API calls (OpenAI/Anthropic), not a local Ollama daemon.
- **LoRA training MUST use query dropout:** `query_dropout_rate=0.5` in `_train_lora_receiver`. Without it, L_sep collapses to 0.0000 and the model ignores the latent entirely. See ARCHITECTURE.md §10 Phase 12.8.
- **Update pipeline is git-hook driven:** `libucks serve` does NOT start WatchdogService. Updates arrive via git post-commit hooks → Unix socket → `StartupRecovery`. For each changed file: if an existing bucket owns it → `UpdateEvent` directly. If no bucket owns it → `StartupRecovery._handle_unmatched_file` does the size+novelty check above and either enqueues a `CreateBucketEvent` or routes to the nearest existing bucket. Use `libucks install-hooks` to wire a target repo.
- **Check sep in training logs:** If `sep=0.0000` persists past epoch 3 in any training run, STOP. The latent is being ignored. Do not continue training — investigate the query dropout code path first.

## 3. Session Start Protocol
When a new session begins:
1. Read `docs/cartridges-log.md` (newest entry) to find the current stage/gate status, and `docs/cartridges-plan.md` for the active roadmap.
2. Scan `tests/unit/` to confirm which tests exist and which are green.
3. Check if `lora_receiver.pt` exists in the target repo's `.libucks/` before assuming LoRA is trained.
4. Never assume training was successful — always verify `sep > 0.0000` in logs.
