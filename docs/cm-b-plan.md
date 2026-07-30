# Plan — CM-B: Living Cartridges

> Continuation of the CM track (CM-0 → CM-A → **CM-B**). Progress is logged in
> `docs/cartridges-log.md`, newest first — deliberately NOT a separate log file, so the
> CM track stays readable end to end.

---

## ⛔ TRACK STATUS 2026-07-31 — THE COMPRESSION LINE IS CLOSED

**Everything below this banner is historical.** Stages 1 and 2 are closed and the
premise of the whole plan — that a compact latent prefix can carry a bucket's content —
has been tested to destruction. Read `docs/cartridges-log.md` CM-B.0i → 0l first;
this file is kept for the reasoning trail, not as a work list.

**Four independent mechanisms failed to make a latent channel carry code facts:**

| mechanism | result |
|---|---|
| Phase 4-C cache augmentation | inert — 2/25, below no-context |
| CM-A cartridge distillation | 6 modern draws at 0–2/8; historical 4/8 and 5/8 never reproduce |
| positional / magnitude selection (CM-B.0i) | score ∝ fraction retained; no compression |
| query-aware selection, SnapKV family (CM-B.0k/0l) | 38% of ceiling at best, non-monotone, no knee |

**And the reframe that matters more than any of them:** with the **entire** bucket in
cache and full attention, the 3B answers **13/26 — half**. That ceiling is verified
three ways (CM-B.0j text-in-prompt control) and reproduces four times. Every
compression number is a fraction of it, so this track was optimising the delivery of
information the reader cannot exploit once it arrives. **The ceiling, not the channel,
is the binding constraint.**

### The one gate that decides whether any of this is worth resuming

**Does the ceiling move with model scale?** Nothing else should be attempted first.

- Run floor + full-cache on the same 26 stratified fixtures at **0.5B, 1.5B, 3B** —
  all three are in the local HF cache, same family and tokenizer, so it is a clean
  comparison and costs nothing but time.
- **Ceiling climbs steeply** → the reader was the limit, compression becomes worth
  revisiting, and every number in the log is re-scoped upward. Then rent a GPU for
  7–8B knowing what to expect.
- **Ceiling flat near 45–50% across a 6× parameter range** → scale is not the lever,
  the limit is the fixtures or the metric, and no further method work is justified
  until that is understood. This is the outcome to expect given how little P mattered.

7–8B is **not runnable locally**: 15.2 GB of bf16 weights against 16 GB of unified
memory, before any cache or transient. Quantising confounds the measurement.

### What is NOT closed, and is where the value actually is

- Routing 1/15 → 14/15 (Phase 1); in-bucket chunk rerank 10 → 16/30 (Phase 3-B);
  **hybrid 19.5 ± 1.7/30** (Phase 4-A, 3-run mean) — the real headline.
- **Raw KV prefill cache**: 13/26 vs a 0/26 floor, ONE forward pass, ~170 MB/bucket.
  Not compression and not a research contribution, but a working feature.
- The self-evolving substrate: mitosis, merge, git-hook updates, novelty-gated spawn.
- **The measurement apparatus**, which is itself an asset: floor harness, position-
  stratified fixtures with enforced anti-bias invariants, the text-in-prompt ceiling
  control, loud-truncation guards. It caught five false positives that would otherwise
  be in a writeup.

### Positioning consequence, stated plainly

Leading with "latent communication" is an overclaim this evidence will not support —
four failed mechanisms, zero surviving positive results, and the literature's own
method reaching 38% where it reports 97–99%. The defensible pillars are the
**self-evolving code memory with measured retrieval gains**, and the **rigorous
negative** on cartridge compression for code QA against a verified ceiling.

## The gap

libucks has two halves that don't compose:

- **Buckets self-evolve.** Mitosis splits them (`mitosis.py:75`), `MergingService` merges
  them (`merging_service.py:120`), `NovelBucketService` spawns them, `HealthMonitor`
  drives it every 5 minutes. This works.
- **Cartridges can't follow.** A cartridge is distilled once from a bucket's corpus. When
  a commit changes that corpus the cartridge is stale, and the only remedy is to throw it
  away and rebuild — **7,199 s per bucket**, measured.

`BucketKVCache` (`bucket_kv_cache.py:35-46`) already hashes every `(chunk_id, git_sha)`
pair, so libucks already **knows** when a cartridge has gone stale. It cannot **repair**
one.

*Mechanism correction (found in the CM-B dependency sweep).* Staleness is detected
**lazily, on read**: `load()` recomputes the chunk signature and returns `None` on
mismatch (`bucket_kv_cache.py:110`), called from `decode.py:70` and
`cache_aug_trainer.py:167`. The eager path — `invalidate()`, whose own docstring says
"call on mitosis / merge / removal" — has **zero callers anywhere in the repo**. An
earlier draft of this plan said the cache is "marked stale on mitosis, merge, or update";
that is wrong about the mechanism, though the effect mostly holds because the next read
detects the change anyway.

Two consequences. Cache files for buckets that mitosis or merge destroyed are never
deleted, since nothing pushes an invalidation — measured as harmless today (0 real
orphans across echoswarm and libugry) only because those events have been rare. And
deliberately **not fixed by wiring `invalidate()` in**: eager deletion on split/merge is
precisely the throw-it-away behaviour Stage 2 exists to replace with repair. Wiring it
now would build the thing this track is trying to remove.

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

**RESULT 2026-07-27: ❌ 1/13 — a regression, not a near miss.** Full entry in
`cartridges-log.md`. The full re-distill is NOT authorised. Headlines:
- The query-gen hypothesis is dead: the control bucket `fe7ded0d` went 2.685 → 2.749 KL,
  i.e. unchanged. And the fix only partly applied — 84/120 and 81/120 model questions,
  the rest silently padded with the templates it was meant to replace.
- KL is decoupled from grounding: `40615ba9` hit **KL 0.227** (better than CM-A.1-retry's
  passing 0.219) and scored **0/3**.
- The run was confounded: it used **120 queries / `max_answer_tokens=32`**, not the
  proven **200 / 48**. That variable has still never been tested at batch scale.

### 0b-repro — faithful CM-A.1-retry reproduction 🔄 RUNNING (2026-07-28)

Before any further hypothesis, establish whether the only passing result in this track
reproduces. `bc6b90e2`, the exact CM-A.1-retry recipe: **200 queries,
`max_answer_tokens=48`**, P=128, 4 epochs, lr 1e-2, verbatim 4096, extract 1024,
**last-epoch save** (matching the original protocol — best-epoch selection is deliberately
deferred so this stays a reproduction, not an improvement).

Chosen over the verbatim/P sweep because `bc6b90e2` carries **8 fixtures** (the best
statistical power of any bucket) and retains 89% of its content, so truncation is not a
confound there.

**Bar:** CM-A.1-retry scored **4/8**; CM-B.0b scored **1/8**. Returning to ~4/8 confirms
the recipe and identifies query count / answer budget as the lever. Staying near 1/8 means
the single passing result never reproduced, and **no downstream plan in this document is
trustworthy until that is resolved** — including Stage 1.

### 0c — cross-model baseline ⬜ TODO, blocks all comparisons

The cartridge runs on 3B (`cm_eval_cartridge.py:32`) while echoswarm's other paths run on
**0.5B** — echoswarm has no `.libucks/config.toml`, so `Config` falls back to the 0.5B
default in `config.py:32-33`. Every cartridge-vs-baseline number is therefore cross-model
and must not be quoted. Fix: pin `base_model = "Qwen/Qwen2.5-3B"` in echoswarm's config
and re-run `no_context` / `text_clean` / `hybrid`. Note echoswarm's `lora_receiver.pt` is
an 896-dim **0.5B** checkpoint and will not load into a 3B receiver.

---

## Stage 1 — the Edit experiment

> **❌ CLOSED 2026-07-29 — the premise no longer holds. Do not build the
> model-backed `TrialRunner`.**
>
> The claim below assumes rebuilding a bucket's memory is expensive (~7,200 s),
> which is what makes cheap repair a research question. CM-B.0i established that
> distillation buys nothing over a raw cache slice at matched size (2/26 vs 1/26),
> and a raw cache is produced by **one forward pass**. When full rebuild costs
> seconds, "can we repair instead of rebuilding?" has no content — the relative
> claim would be trivially true and uninteresting.
>
> Closed because the problem dissolved, not because the experiment failed. The
> scaffolding (`cm_make_edits.py`, `cartridge_edit.py`, `cm_edit_experiment.py`,
> 138 tests) stays in the tree: the edit-generation and staleness-detection halves
> are reusable if a working compressor ever makes per-bucket artifacts expensive
> again.

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

> **❌ CLOSED 2026-07-31.** Was contingent on Stage 1, which closed 2026-07-29; and the
> compression premise underneath both is now closed by CM-B.0l. Cartridge split/merge
> only matters if a cartridge carries content worth preserving across a mitosis, and it
> does not — 2/26 against a 0/26 floor and a 13/26 ceiling. Do not build this.

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

## Dependency-graph sweep — structural findings

Run over `libucks/` + `scripts/` + `tests/` (import graph, cycles, layering, dead code,
config-key readers). Architecture is clean: **no runtime import cycles, no layering
violations** (nothing low-level imports upward), and every production module except
`_cli.py` (2,035 loc, untested) has at least one test.

What it did surface, none of it a live crash:

| finding | status |
|---|---|
| Merge limit hardcoded above a tunable split threshold | **fixed** — derived, `12c1af7` |
| Six `_encode_centroid` copies, one decoder | **fixed** — consolidated, `12c1af7` |
| HealthMonitor re-embedded every chunk every 300 s | **fixed** — cached, `e622585` |
| `chunk_retriever` used outside its nested scope | **fixed** before running — `NameError` at serve time, `e622585` |
| `margin_separation_loss` never called, never tested | documented in `losses.py` |
| `alignment_loss` imported, never called | leave |
| `BucketKVCache.invalidate()` zero callers | documented above; not wiring, on purpose |
| `compression_steps` inert config knob | documented in `config.py` + RUNBOOK |
| `PathsConfig.{grammar_cache, log_file, pending_events, repo_cache}` never read | inert; documented in ARCHITECTURE.md as if live |
| `_inject_lora_into_module_dict`, `_pool` dead | leave |

The `margin_separation_loss` one is the most interesting for this project: it was written
specifically to escape the `sep=0.0000` collapse CLAUDE.md tells you to halt on, and it
has never been called in the entire git history. See its docstring.

## House rules

Tests first, watch them fail, then implement. No advancing on a red gate. Every stage's
numbers and verdict appended to `docs/cartridges-log.md` before moving on. Every distill
or eval run needs `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` before the torch import and must
be `nohup`-detached under `caffeinate -dimsu`. HF `generate()` aborts the process on MPS
for long prompts — run generation on CPU, or use a manual decode loop.
