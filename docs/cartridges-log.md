# Cartridge Memory (CM) — Progress log

Append-only log; one section per sub-stage. Newest entries at the top.
The plan being executed is `docs/cartridges-plan.md`. If the plan changes
mid-execution, add a "Plan revision" entry noting why.

Prior tracks (V1, V2, Phase 3/4-A/4-C) are archived under `docs/archive/`.
Phase 4-C closed as a **negative result** (latent channel inert); its full
log is at `docs/archive/phase-4c/phase-4c-log.md`. CM starts fresh from that
finding.

## Table of contents

- CM-0 — Documentation cleanup & archive — DONE 2026-07-01 (root .md 11→6; papers consolidated to docs/papers/; phase-4c + V1/V2 plans archived)
- CM-A.0 — Scaffold + cartridge contract (TDD) — DONE 2026-07-01 (6/6 KVPrefixCartridge contract tests green; module landed)
- CM-A.1 — Single-bucket distill proof — ⚠ GATE FAIL 2026-07-02 (KL 0.916→0.517; latent-alone grounding 1/8→2/8, needed ≥3; objective works but latent carries structure not identifiers — Phase-4A decomposition reproduced under distillation)

---

## CM-A.1 — Single-bucket distill proof (2026-07-02)

**Status**: ⚠ GATE FAIL — informative. Objective works; capacity/data-coverage limited.

**Setup**: echoswarm bucket `bc6b90e2` (agents/relay, 8 routed fixtures, 948-token
verbatim). Warm-start KVPrefixCartridge (P=64) from real extracted KV; distill with
context distillation (KL + 0.3·CE) on 128 **templated** self-study queries, 2 epochs,
lr=1e-2, frozen Qwen2.5-3B on MPS. Script: `scripts/cm_proof_single_bucket.py`.

**Result**:
- KL 0.916 → 0.517 (−44%, still falling at epoch 2 end).
- Latent-alone grounding: warm-start **1/8** → distilled **2/8** (gate needed ≥3).
- Distilled answers are coherent + on-topic but miss specific identifiers
  (e.g. "relay probability is 0.5" — truth is 80%; missed 0.6, STRANDED, "2 confirmations").

**Finding**: context distillation is the right *mechanism* (KL dropped, grounding rose,
fluency improved vs. the cosine objective's noise), but at this scale the latent carries
**structure, not precise identifiers** — the Phase-4A two-channel decomposition, reproduced
under a proper objective. This is the load-bearing honest result.

**Suspected causes (ranked)**:
1. Query coverage — templated queries don't force the teacher to state the tested facts
   (if "80%" never appears in training answers, the cartridge can't learn it).
2. Prefix capacity P=64 for a 948-token bucket.
3. Under-trained — 2 epochs, KL still falling.
4. Warm-start from first-64 KV biases to the file header, not the fact-dense body.

**Gate result**: FAIL (2/8 < 3/8). Decision pending (see chat): bounded lever test
(model-generated + fact-probing queries, P=128–256, +epochs) on the same bucket, vs.
bank as negative result.

**Next**: user decision.

## CM-A.0 — Scaffold + cartridge contract (2026-07-01)

**Status**: ✅ gate passed.

**Built**:
- `tests/unit/test_cartridge.py` — TDD contract for `KVPrefixCartridge` (shapes,
  init-from-extracted-KV incl. short-bucket pad, grad-flow, save/load roundtrip).
  Written first; watched it fail (ModuleNotFoundError).
- `libucks/cache_augmentation/cartridge.py` — `KVPrefixCartridge(nn.Module)`:
  per-layer trainable `(1, n_kv_heads, P, head_dim)` K/V ParameterLists;
  `init_from_extracted_kv` (warm-start from `kv_extract` flat dict, first-P copy
  with short-bucket slack); `to_dynamic_cache` (grad-preserving DynamicCache);
  safetensors `save`/`load`.

**Gate result**: `uv run pytest tests/unit/test_cartridge.py` → **6/6 green**.

**Next**: CM-A.1 — `distillation_loss` (KL to full-context teacher), `self_study`
query generation, `cartridge_trainer` (backprop into prefix only, base frozen),
real-model integration smoke, then single-bucket distill proof.

## CM-0 — Documentation cleanup & archive (2026-07-01)

**Status**: ✅ done.

**Done**:
- Created `docs/archive/` and `docs/archive/phase-4c/`; archived (via `git mv`,
  history preserved): `IMPLEMENTATION_PLAN.md`, `V2_IMPLEMENTATION_PLAN.md`,
  old `QUICKSTART.md` (V1), `INTERLAT_LITE_MANUAL.md` (Phase-12 runbook, stale),
  `LATENT_RAG_ARCHITECTURE.md` (design record of the now-superseded latent LoRA
  approach), and `docs/phase-4c-{plan,log}.md`.
- Promoted `QUICKSTART_V2.md` → `QUICKSTART.md` (single current quickstart).
- Consolidated 15 scattered arXiv paper markdowns (`docs/*.md` + `docs/new-docs/`)
  into `docs/papers/`; deduped the `2412.06769` copy; removed empty `docs/new-docs/`.
- Repointed `CLAUDE.md` §1 + §3 from `IMPLEMENTATION_PLAN.md` to
  `docs/cartridges-plan.md` / `docs/cartridges-log.md`; updated the `ARCHITECTURE.md`
  repo-tree line.
- Kept active at root: `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`, `POCSTRAT.md`,
  `PITCH.md`, `QUICKSTART.md`.

**Next**: CM-A.0 — scaffold smoke test + begin the cartridge implementation.

---

## Entry template (copy for each new sub-stage)

```
## CM-X — <name> (<date>)

**Status**: ✅ gate passed / ⚠ partial / ❌ blocked
**Built**: <bullets>
**Decided**: <bullets>
**Issues**: <bullets>
**Gate result**: <criterion>; <met / not met>
**Next**: CM-X+1 — <name>
```
