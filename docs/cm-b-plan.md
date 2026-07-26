# Plan — CM-B: Living Cartridges

> Continuation of the CM track (CM-0 → CM-A → **CM-B**). Progress is logged in
> `docs/cartridges-log.md`, newest first — deliberately NOT a separate log file, so the
> CM track stays readable end to end.

## The gap

libucks has two halves that don't compose:

- **Buckets self-evolve.** Mitosis splits them (`mitosis.py:75`), `MergingService` merges
  them (`merging_service.py:120`), `NovelBucketService` spawns them, `HealthMonitor`
  drives it every 5 minutes. This works.
- **Cartridges can't follow.** A cartridge is distilled once from a bucket's corpus. When
  a commit changes that corpus the cartridge is stale, and the only remedy is to throw it
  away and rebuild — **7,199 s per bucket**, measured.

`BucketKVCache` (`bucket_kv_cache.py:35-46`) already hashes every `(chunk_id, git_sha)`
pair and marks the cache stale on mitosis, merge, or update. libucks already **knows**
when a cartridge has gone stale. It cannot **repair** one.

## Why this is worth doing

[Cartridges at Scale](https://arxiv.org/abs/2606.04557) (2026), the current SOTA, states
the gap in its own words — *"updating or adding a single document requires re-encoding the
entire KV cache"* — and does not solve it. Its answer is finer granularity, not repair.
It also leaves grouping (split/merge) explicitly to future work. Nobody edits a trained
cartridge in place.

Adjacent but different: [Language Models Need Sleep](https://arxiv.org/pdf/2606.03979)
consolidates **weights**; [RefreshKV](https://arxiv.org/abs/2411.05787) updates a cache
**during generation**; [C²KV](https://arxiv.org/html/2607.17715) concatenates caches for
serving throughput.

**The question is not "is latent better?" — it is: when a bucket's code changes, can its
cartridge be updated in place instead of rebuilt from nothing?**

That is a *relative* claim (cheap repair vs full rebuild, both inside libucks), not an
*absolute* one (latent beats RAG). Absolute claims need 7–8B models and have already
failed twice here. Relative claims are laptop-scale and measured in KL, which carries none
of the decode-loop variance that produced Phase 4-C's phantom win.

Repos are the right substrate, not sunk cost: **git supplies free, versioned,
ground-truth deltas.** No other memory domain has a principled notion of "chunk c3 became
c3′".

**Why it is research and not plumbing:** a cartridge has no index. In a text bucket you
edit line 47. In a cartridge, whatever was learned about chunk c3 is smeared across 2.36M
numbers with no map back to source. Whether that is localizable at all is the crux.

---

## Stage 0 — verify the baseline (in progress)

### 0a — grounding metric audit ✅ DONE 2026-07-27

Cartridge **7/25 → 10/25**, gate ≥8/25 **PASS**. Baselines moved ≤ +1, so not inflation.
Full detail in `docs/cartridges-log.md`. Landed `libucks/eval_metrics.py` as the single
shared scorer; both `test_latent_vs_baseline.py` and `cm_eval_cartridge.py` delegate to it.

### 0b — query-gen regression fix ⏳ RUNNING

`cm_distill_buckets.py:96` passed `model=None`, which skips `_model_queries` and falls
through to templates (`self_study.py:178-185`) — the configuration CM-A.1 had already
shown to fail (2/8 templated vs 4/8 fact-probing). Fixed; first bucket now yields 112
model-written questions where CM-A.2 had 0.

Spot-check: `bc6b90e2` (2/8), `40615ba9` (0/3), `fe7ded0d` (0/2 — the control; already
cleanly re-distilled 6× under the old config, so a win here isolates the query variable).

**Bar:** those 13 fixtures score 5/13 under the CM-B.0a metric. `> 5/13` justifies the
full ~20 h re-distill of all ten buckets.

### 0c — cross-model baseline ⬜ TODO, blocks all comparisons

The cartridge runs on 3B (`cm_eval_cartridge.py:32`) while echoswarm's other paths run on
**0.5B** — echoswarm has no `.libucks/config.toml`, so `Config` falls back to the 0.5B
default in `config.py:32-33`. Every cartridge-vs-baseline number is therefore cross-model
and must not be quoted. Fix: pin `base_model = "Qwen/Qwen2.5-3B"` in echoswarm's config
and re-run `no_context` / `text_clean` / `hybrid`. Note echoswarm's `lora_receiver.pt` is
an 896-dim **0.5B** checkpoint and will not load into a 3B receiver.

---

## Stage 1 — the Edit experiment

**Claim:** when one chunk changes, a cheap warm-started repair lands close to a full
re-distill at a fraction of the cost.

Single-bucket, so it does **not** depend on Stage 0 — CM-A.1 already proved single-bucket
cartridges work (4/8, KL 0.749 → 0.219).

1. **Controlled edits** (`scripts/cm_make_edits.py`) — rename, constant change, added
   branch, deleted helper, in increasing order of disturbance. Synthetic rather than real
   history because libugry has 1 commit and echoswarm's 14 are nearly all README edits —
   and controlling edit size is what produces the repair-cost-vs-edit-magnitude curve,
   which is the result.
2. **Ground truth** — one full re-distill per edit (~7,200 s), the correct answer and the
   cost baseline.
3. **Repair methods** (`libucks/cache_augmentation/cartridge_edit.py`):
   - *Continue-training* — warm-start from the existing cartridge, re-distill only on
     queries whose teacher answers changed. Reuses `CartridgeTrainer._step_cached`
     (`cartridge_trainer.py:200`), a single teacher-forced forward instead of greedy
     generation. The honest baseline.
   - *Slot-localized* — `init_from_extracted_kv` (`cartridge.py:73`) warm-starts by
     copying the first P positions of the bucket's real KV, so a positional
     correspondence to source exists before training. Test whether any survives
     distillation; if so, retrain only affected slots. **This is the research bet.**
   - *Low-rank delta* — freeze the cartridge, learn an additive correction. Fallback.
4. **Metrics** — primary `KL(repaired ‖ re-distilled)` (no generation, no decode
   variance); headline `cost ratio`; secondary grounding delta; and the **staleness
   floor** (score of the un-repaired cartridge) to prove the edit disturbed anything at
   all.

**Pre-registered gate:** within 10% of full-re-distill KL at ≤25% of the cost, on ≥3 of
the 4 edit types. Fixed before running.

**Pre-committed honest outcome:** if repair is trivially easy because cartridges barely
react to edits, that is *insensitivity*, not a mechanism. Publish it as such.

---

## Stage 2 — split and merge (contingent)

Only if Stage 1 clears. Can a cartridge split when mitosis splits its bucket? Can two
compose when buckets merge? Merge is the hard one — CAS shows naive concatenation
collapses to near-chance, which is very likely what CM-A.2's multi-bucket top-k concat
(`cartridges-plan.md:106`) was also hitting.

## Known measurement fragilities

Found in the CM-B.0b bug sweep. Neither is fixed, both are deliberate deferrals.

**Self-study covers only 73% of each bucket.** `_model_queries` caps its context at
`bucket_text[:3000]` (`self_study.py:98`) while echoswarm buckets are ~4,126 chars — so
roughly 27% of every bucket has no training question written about it. CM-A.1's finding
was literally "query coverage was the bottleneck", so this is on the critical path.

*Not fixed mid-run on purpose.* The Stage 0b run is testing exactly one variable —
fact-probing vs templated queries. Raising coverage at the same time would change two
things at once and make the result uninterpretable. Fix it for the full ten-bucket
re-distill, where it becomes a clean second improvement.

**Teacher Q&A sees only 1024 chars in three of four call sites.** `_cli.py` has
four sibling calls that build the source text handed to the Anthropic teacher.
One (`:1690`, in `generate-qa`) was raised to `max_chars=4096` with a comment
explaining why 1024 was wrong. The other three — `:684` and `:742` in
`train-adapter`, and `:1730`, the `--no-teacher` branch of `generate-qa` itself
— were never updated and still pass 1024. Echoswarm buckets average ~4,126
chars, so those paths see roughly a quarter of each bucket.

The `--no-teacher` branches are the worse case: they store the truncated text
*as the training target* (`"pairs": [[PERSPECTIVE_PROMPTS[0], src]]`), so the
LoRA receiver is trained to reproduce a quarter-bucket.

*Not fixed.* `train-adapter` produces `lora_receiver.pt`, which the hybrid path
— the Phase 4-A headline of 19.5 ± 1.7/30 — depends on. Changing its training
inputs invalidates comparisons against every existing checkpoint, so this is a
deliberate decision to make with a re-train, not a drive-by edit. The
divergence is now flagged in-place at `_cli.py:1690`.

Note the original justification is itself stale: the comment describes
`_collect_source_text` returning `""` outright on overflow, which was fixed on
2026-07-03 (it now slices). The failure mode changed from "blanks ~25% of
buckets" to "truncates all of them".

**Three fixtures route within 0.01 of a tie.** Fixture→bucket routing is top-1 cosine over
centroids. It is deterministic within a run and verified byte-identical to CM-A.2's stored
routing (0 of 25 changed), so the 5/13 spot-check baseline is valid. But 3 of 25 fixtures
have a top1–top2 margin below 0.01, so any centroid recomputation, mitosis/merge, or
embedding-model change can silently flip which cartridge they are scored against. Before
trusting any future cross-run comparison, re-run the routing-stability check.

## House rules

Tests first, watch them fail, then implement. No advancing on a red gate. Every stage's
numbers and verdict appended to `docs/cartridges-log.md` before moving on. Every distill
or eval run needs `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before the torch import and must
be `nohup`-detached under `caffeinate -dimsu`. HF `generate()` aborts the process on MPS
for long prompts — run generation on CPU, or use a manual decode loop.
