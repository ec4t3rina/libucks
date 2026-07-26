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
- CM-A.1 — Single-bucket distill proof — ⚠ GATE FAIL 2026-07-02 (v1: templated queries, P=64, 2ep → 2/8)
- CM-A.1-retry — Fact-probing queries + P=128 + 4ep — ✅ GATE PASS 2026-07-02 (**latent-alone 2/8→4/8; KL 0.749→0.219; carries identifiers — "80%", garble, STRANDED. Hypothesis confirmed: query coverage was the bottleneck**)
- CM-A.2 — All-bucket distill + eval gate — ❌ GATE FAIL 2026-07-07 (latent-alone 7/25 vs gate ≥8, bit-identical across 2 evals; fe7ded0d redistill r6 clean (KL 3.69→2.69) but converts neither of its fixtures; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` proven causally necessary) — **SUPERSEDED by CM-B.0a: the 7/25 was a metric artifact; re-scores to 10/25 = GATE PASS. Also ran templated queries, not the fact-probing ones the entry claims.**
- CM-B.0a — Grounding metric audit + full re-score — ✅ GATE PASS 2026-07-27 (**cartridge latent-alone 7/25 → 10/25 vs gate ≥8; baselines move ≤+1, so not general inflation; CM-A.2's negative result is overturned**)

---

## CM-B.0a — Grounding metric audit + full re-score (2026-07-27)

**Status**: ✅ GATE PASS — CM-A.2's ❌ is overturned. Zero compute; scoring only.

**Found**: `_grounding_score` (`test_latent_vs_baseline.py:451`) did plain
case-insensitive substring matching, so a correct answer in a different surface form
scored as wrong. Three of CM-A.2's 18 "failures" were correct:

| fixture | expected | model said | |
|---|---|---|---|
| echoswarm_01 | `80%` | "relay probability … is **0.8**" | correct, scored wrong |
| echoswarm_02 | `2` | "at least **two** different sources" | correct, scored wrong |
| echoswarm_03 | `1` | "randomly changing **one** character" | correct, scored wrong |

**Built**:
- `libucks/eval_metrics.py` — `keyword_variants` / `keyword_hit` / `grounding_score`.
  Literal keyword keeps plain substring matching (so no previously-passing fixture can
  start failing, and historical numbers stay comparable); *added* variants
  (percent↔decimal, number word↔digit) match on word boundaries only — without that,
  keyword "1" → variant "one" would fire inside "money".
- `tests/unit/test_eval_metrics.py` — 19 tests, written first, watched fail
  (ModuleNotFoundError). Includes the four real CM-A.2 cases as regressions: 01/02 must
  flip to grounded, 06/10 must stay failures.
- `scripts/cm_rescore_grounding.py` — re-scores stored answers under both metrics.
- `test_latent_vs_baseline.py:451` now delegates to the shared function.

**Gate result** (`uv run python scripts/cm_rescore_grounding.py`):

| path | old | new | Δ |
|---|---|---|---|
| **cartridge (CM-A.2)** | **7/25** | **10/25** | **+3** |
| cache_aug_no_verbatim | 2/25 | 3/25 | +1 |
| hybrid | 11/25 | 11/25 | 0 |
| text_clean | 4/25 | 4/25 | 0 |
| no_context | 3/25 | 3/25 | 0 |
| latent | 2/25 | 2/25 | 0 |
| cache_aug | 12/25 | 12/25 | 0 |

Old metric reproduces the logged 7/25 exactly, so the re-score is trustworthy.
**Gate ≥8/25: FAIL → PASS.** The correction is not general inflation — every baseline
moves by ≤+1, because the baselines' failures are vague or wrong answers, not
correctly-stated facts in another format.

**Issues / not yet resolved**:
- **The 10/25 vs 11/25 hybrid comparison is cross-model and must not be quoted.** The
  cartridge ran on 3B (`cm_eval_cartridge.py:32`); hybrid / text_clean / no_context ran
  on 0.5B (echoswarm has no `.libucks/config.toml`, so `Config` falls back to the 0.5B
  default). A 3B no_context / text_clean baseline is still owed.
- Phase 4-C stays a negative result: `cache_aug_no_verbatim` re-scores to 3/25, exactly
  the no_context floor.
- CM-A.2 additionally ran **templated** queries — `cm_distill_buckets.py:96` passes
  `model=None`, which skips `_model_queries` entirely (`self_study.py:178-185`) — despite
  the entry claiming fact-probing self-study. CM-A.1 had already shown templates were the
  bottleneck (2/8 → 4/8). So 10/25 is the *templated-query* score; the fact-probing
  re-run (CM-B.0b) is still owed and should go higher.

**Next**: CM-B.0b — fix `cm_distill_buckets.py:96`, re-distill bc6b90e2 / 40615ba9 /
fe7ded0d, re-eval those 13 fixtures (currently 2/13 old, 5/13 new-metric).

## CM-A.2 — All-bucket distill + eval gate (2026-07-07)

**Status**: ~~❌ GATE FAIL, reproduced — latent-alone 7/25 vs gate ≥8/25~~ — **SUPERSEDED
2026-07-27 by CM-B.0a.** The 7/25 was a scoring artifact: three of the 18 "failures" were
correct answers in a different surface form ("0.8" vs `80%`, "two" vs `2`, "one" vs `1`).
Re-scores to **10/25 = GATE PASS**. Two further corrections to the entry below: it states
"fact-probing self-study", but `cm_distill_buckets.py:96` passed `model=None` and therefore
ran **templated** queries; and the reference baselines it quotes were measured on a 0.5B
receiver while the cartridge ran on 3B. Read the numbers below as the record of what was
run, not as a valid gate result.

**Setup**: all 10 fixture-routed echoswarm buckets distilled (fact-probing self-study,
N=120 queries, P=128, 4 epochs, lr=1e-2, frozen Qwen2.5-3B/MPS) via
`scripts/cm_distill_buckets.py` (resumable batch; per-epoch checkpoints). Eval:
`scripts/cm_eval_cartridge.py`, 25 fixtures, latent-alone (no verbatim).

**The fe7ded0d redistill (6 attempts, Jul 3–7)** — its original cartridge predated the
`_collect_source_text` truncation fix (30ee434) and was deleted for redistill:
- r1 hung mid-step early in epoch 1; r2 reached ep1 (KL 3.97→2.84) then wedged in
  `torch.mps.empty_cache()` mid-ep2 (uninterruptible MPS wait); r3 hung in teacher
  precompute alongside the zombie r2.
- r4 (launched WITHOUT `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`) wedged at the first
  teacher-generate. A standalone clean-process probe wedged without the env var and
  passed with it (62 s) → **the env var is causally NECESSARY on this machine for any
  distill/eval run**. Pressure-sensitive (16 GB RAM), which is why Jul 2–3 runs
  sometimes passed without it.
- r5 (env var on) exited cleanly at 01:05 mid-precompute (40/120) — launched
  non-detached, killed when its parent session closed.
- r6 (env var on, `nohup caffeinate -dimsu`, detached): **completed** —
  KL 3.690 → 2.685 (4 ep, one recovered bump at ep2: 2.84→3.08→2.69), 7199 s.

**Result** (eval r2, new fe7ded0d cartridge; eval r1 preserved as
`tests/eval/results/cm/echoswarm_cartridge_A2_r1_fail7.json`):
- Latent-alone grounding **7/25** (multi 3/7) — **bit-identical fixture set to eval r1**
  (which ran with the old broken fe7ded0d cartridge). Stable result, not variance.
  Gate ≥8/25 → **FAIL**.
- Refs (eval r1): hybrid 11/25, text_clean 4/25, no_context 3/25,
  cache_aug_no_verbatim 2/25.
- The redistilled fe7ded0d is qualitatively better on its two fixtures but converts
  neither: echoswarm_10 went degenerate loop ("the CER of the CER of…") → fluent
  three-pillar CERC structure with WRONG expansions ("Crisis, Economic, Reputation"
  vs FRAMING/CLARITY/CONTENT); echoswarm_16 still misses all 5 highway-class keywords.
  **Structure, not identifiers** — the CM-A.1-v1 failure mode at bucket scale.
- No near-misses anywhere: best failed fixtures carry 1 of 3–5 required keywords;
  10 of 18 carry 0. There is no cheap +1.
- Honest positive: 7/25 is **+4 over no_context (3/25)** with multi 3/7 — the cartridge
  channel is *alive*, unlike Phase 4-C cache-aug (2/25, below floor). The fail is
  "under the strong-win bar", not "inert".

**Anomaly for the decision**: the batch used **N_QUERIES=120**, below the proven
CM-A.1-retry recipe (**200** queries on bc6b90e2 → 4/8); in this eval bc6b90e2 scores
2/8 (grading harnesses differ, so directionally suggestive only). The plan
pre-authorizes scaling self-study to 512–1000 when the gate is borderline —
one-short-twice qualifies.

**Gate result**: latent-alone ≥8/25; **not met (7/25, twice)**.

**Next**: user decision — bounded query-scale retry (512 queries on the high-miss
buckets bc6b90e2 / 40615ba9 / 95c8e099, ~4 h/bucket, needs +1 conversion) vs bank as
second negative result ("3B is the ceiling for latent-alone identifiers").

## CM-A.1-retry — Fact-probing queries + P=128 + 4 epochs (2026-07-02)

**Status**: ✅ GATE PASS.

**Changed vs v1** (bounded lever retry, same bucket `bc6b90e2`):
- Self-study queries: generic templates → **fact-probing templates** (force the teacher
  to state specific values: numbers, probabilities, thresholds, states). This was the #1
  suspected cause and it was correct.
- Prefix P=64 → **128**; epochs 2 → **4**; queries 128 → **200**.

**Two bugs fixed to get here** (both real, both in `cartridge_trainer.py`):
1. **Missing per-step `torch.mps.empty_cache()`** → MPS memory fragmented and the process
   HUNG at ~step 180 (observed twice). cache_aug_trainer does per-step empty_cache; mirrored.
2. **Teacher regeneration every epoch** → the frozen+deterministic teacher answer was
   greedy-regenerated (48 seq forwards) every step every epoch. Refactored to **precompute
   answers once**, then teacher logits via a **single teacher-forced forward** per step.
   Cut projected runtime from ~2h to ~30 min. Validated by smoke (KL still 5.66→4.21→3.61).

**Result** (echoswarm bucket `bc6b90e2`, 8 fixtures, latent-alone / no verbatim):
- KL 0.749 → 0.219 (4 epochs, monotonic, −71%).
- Latent-alone grounding: warm-start **2/8** → distilled **4/8** (gate ≥3 and > init) → **PASS**.
- Carries specific identifiers now: "relay probability is **80%**" (v1 said 0.5), Panic
  garble, Immobile→"never receive", INFORMED preservation — all from the latent alone.

**Finding**: context distillation + **fact-probing self-study** makes the latent channel
carry facts. The whole-project failure mode (latent ≈ no_context; Phase-4A 2/25, cache-aug
no-verbatim 2/25) is broken here: 4/8 latent-alone on a real uncontaminated bucket. The
prior gate fail was query coverage, not a fundamental ceiling.

**Caveats (honest)**: n=1 bucket, 8 fixtures; 3B/MPS; fact-probing queries are templated
(not model-generated — MPS crashed on 0.5B generate, CPU too slow). CM-A.2 must confirm
this holds across all 33 buckets (gate: latent-alone ≥ 8/25 vs no_context 3/25).

**Next**: CM-A.2 — all-bucket distill + full 5-path eval.

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
