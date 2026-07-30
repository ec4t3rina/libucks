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
- CM-B.0b — Query-gen fix + 3-bucket re-distill — ❌ GATE FAIL 2026-07-27 (**1/13 vs a 5/13 baseline — a regression, not a near miss. bc6b90e2 5/8→1/8 with 6/13 answers degenerate token loops. Confounded: still 120 queries, not the proven 200. Diagnostics found the verbatim cap discards 66–72% of two buckets and the eval ceiling is 20/25, not 25/25**)
- CM-B.0b-repro — CM-A.1-retry reproduction, 3 seeded draws — ❌ DOES NOT REPRODUCE 2026-07-28 (**2/8, 1/8, 1/8 — spread 1, so this is NOT noise**)
- CM-B.0d — CM-A.2's exact config, 3 seeded draws — ❌ DOES NOT REPRODUCE 2026-07-28 (**0/8, 1/8, 1/8. CM-A.2's 5/8 was a single lucky draw. Six modern draws across two configs all land at 0–2/8 against two historical single draws of 4/8 and 5/8**)
- CM-B.0e — no-context floor, 3 arms × 3 seeds — ❌ **LATENT CHANNEL INERT AT P=128** 2026-07-28 (**cartridge − floor = +0.67/8 and +0.33/16, i.e. zero at spread ±1. On the non-leaky set the base 3B scores 0.00/8 cold and the cartridge scores 0.67/8. A random prefix is ACTIVELY HARMFUL (2.33 vs floor 4.00), so `cartridge − random` overstates the benefit and must not be quoted**)
- CM-B.0f — the two surviving objections: best-epoch and P — ⚠️ **P WAS A CONSTRAINT** 2026-07-29 (**P=384 gives c−floor +3/8 and +5/16 vs +0.67 and +0.33 at P=128, and initial KL 5.68→3.63 from capacity alone. The CM-B.0e "inert" verdict was a P=128 artifact — but see 0g/0h: P was not the *binding* constraint**)
- CM-B.0g — P sweep, 384×3 seeds + 512 + 768 — ⚠️ **KL AND GROUNDING DECOUPLE HARD** 2026-07-29 (**P=768 reaches the best KL ever recorded here (1.093) and scores 2/8 — worse than P=512's 4/8 on a 3× worse KL. Threefold divergence improvement buys nothing. Diagnosis: at high P the cartridge fits the 120 self-study queries, and the fixtures ask different questions**)
- CM-B.0h — training-free KV-cache selection — ⚠️ **CONCLUSION RETRACTED by CM-B.0i** 2026-07-29 (**measured `kv_first` 13/16 vs cartridge 8/16 and concluded distillation degrades its own warm start. Both the fixture set and the conclusion were flawed: the bc6b90e2 fixtures cluster in the first quarter of the file, and `kv_first` keeps the FIRST P positions, so it won by construction. On position-stratified fixtures the two are indistinguishable**)
- CM-B.0i — position-stratified sweep on a second bucket — ✅ **TRUNCATION DOES NOT COMPRESS; UNCOMPRESSED CEILING IS 50%** 2026-07-29 (**full 4,599-token cache scores 13/26 vs floor 0/26. Score is proportional to fraction retained — 2.8%→1/26, 22%→6/26, 100%→13/26 — so no prefix carries information about text it dropped. At matched P=128 cartridge 2/26 ≈ kv_first 1/26. Both CM-B.0h/0i-precursor headlines were artifacts of self-authored, head-clustered fixtures**)
- CM-B.0j — text-in-prompt ceiling control + MPS memory profile — ✅ **THE 0i CEILING IS REAL, NOT A HARNESS ARTIFACT** 2026-07-30 (**same text as prompt tokens read with full attention scores 14/26 vs the serialised full cache's 13/26. The +1 is decode noise, not a lossy cache path: the 5 disagreements are BIDIRECTIONAL (3 text-wins, 2 cache-wins). 0i reproduces exactly — 13/26 and cartridge 2/26 — on different code paths. So `1/26 at 36×` is 8% of a REAL ceiling and the negative result stands. Also: `generate_answer` truncated silently at 3,500 tokens and would have produced a FALSE confirmation; the same cap means every cartridge here was distilled from a teacher that saw only 76% of its bucket. MPS profile: no leak of any kind, but one 4,599-token prefill permanently caches 4.5 GB that `empty_cache()` will not return**)
- CM-B.0k — query-aware KV selection (SnapKV family) — ⚠️ **BEST LEAD IN THIS TRACK, NOT ESTABLISHED** 2026-07-30 (**at P=128 / 35.9×: query-aware per-layer selection 5/26 vs `kv_first` 1/26, `kv_norm` 0/26 and the 98-minute distilled cartridge 2/26, against a 13/26 ceiling. 5× the positional selector and +3 over training, with ZERO training — but exact McNemar on the 6 discordant pairs gives p = 0.219, so NOT significant. A bit-identical repeat (0/182 verdicts and 0/182 answer texts differ) proves the pipeline is deterministic and therefore CANNOT add evidence for the effect. Also 38.5% of ceiling where the literature reports 97–99%, and it buys attention-COMPUTE not STORAGE compression. Needs ~8–1 or 6–0 discordant to clear p<0.05: more fixtures, a P sweep, or per-head+pooling**)

---

## CM-B.0k — query-aware KV selection (2026-07-30)

**Status**: ⚠️ The best lead this track has produced, and **not established**. The
effect size is large in ratio terms; the significance is not there. Both facts below.

**The gap this closes.** CM-B.0i concluded no compression mechanism exists, from a
sweep in which *every selector was query-agnostic*: `kv_first`/`kv_last`/`kv_stride`
are positional and `kv_norm` ranks by ‖K‖ — described in its own source comment as
"a cheap stand-in for attention importance that needs no second forward pass." The
literature does not use a stand-in. SnapKV / H2O / CompressKV keep the positions the
ACTUAL QUERY attends to. So 0i's negative was never a statement about the content;
it was a statement about the selector, and that was never tested.

### Result — P=128, 35.9× compression, n=26, ceiling 13/26

| arm | score | % of ceiling | |
|---|---|---|---|
| floor | 0/26 | 0% | |
| `kv_norm@128` | 0/26 | 0% | query-agnostic, magnitude |
| `kv_first@128` | 1/26 | 7.7% | query-agnostic, positional |
| distilled cartridge | 2/26 | 15.4% | 98 minutes of training |
| `kv_attn@128` | 3/26 | 23.1% | **query-aware, global** |
| **`kv_attn_L@128`** | **5/26** | **38.5%** | **query-aware, per-layer** |
| full cache | 13/26 | 100% | the ceiling, 1.0× |

`+4` over `kv_first`, `+5` over `kv_norm`, `+3` over the cartridge — with **no
training at all**, against 98 minutes of distillation.

### Significance — the part that does not hold up

The first draft of this entry argued the result was signal because the per-fixture
wins are *asymmetric*: query-aware takes y01, y04, y19, y20, y23 where `kv_first`
fails and loses only y06, i.e. 5–1 directional, versus CM-B.0j's 3–2 bidirectional
split that was correctly dismissed as noise. Direction genuinely is the right
discriminator for paired binary outcomes. **But the correct test was never run, and
it does not pass:**

| comparison | wins–losses | discordant | exact two-sided McNemar |
|---|---|---|---|
| `kv_attn_L` vs `kv_first` | 5–1 | 6 | **p = 0.219** |
| `kv_attn` (global) vs `kv_first` | 3–1 | 4 | p = 0.625 |
| `kv_attn_L` vs cartridge | 3–0 | 3 | p = 0.250 |

Six discordant pairs splitting 5–1 is well inside what a fair coin produces. The 5×
ratio and the +4 absolute are real numbers, but they do not clear a significance bar,
and "believed to be signal" was the wrong claim to write down.

**The repeat cannot help, and this is the useful part of running it.** A second run is
**bit-identical**: 0/182 grounded verdicts and 0/182 generated answer texts differ,
with the setup memory line matching byte-for-byte. Cache building, scoring and top-p
are deterministic and the decode is greedy, so there is no seed here to vary. The
pipeline is reproducible — which rules out flakiness and is worth having — but a
deterministic experiment re-run adds exactly zero independent evidence about effect
size. (This also contradicts the ±2 MPS nondeterminism recorded from earlier work;
for this code path there is none. Possibly eager attention is more stable than SDPA.)

**What would actually clear p < 0.05**, given the exact test: 6–0 (p = 0.031), 8–1
(p = 0.035) or 9–1 (p = 0.020). From 5–1 that is roughly **three more net wins** —
reachable only by more fixtures, more budgets, or a stronger selector, never by
repetition.

**Two things that do hold.** Per-layer beats global 5 vs 3, matching SnapKV's stated
reason for selecting per layer — weak evidence, but consistent with a prior rather
than tuned toward. And the yardstick held for the third time: full cache 13/26 and
cartridge 2/26 are identical across CM-B.0i, 0j and 0k on three different code paths.

**Process note, second occurrence tonight.** Earlier in the same session I reported
"11/11 agreement, zero disagreements" at a partial tally that finished at 21/26, and
here I called 5–1 signal before testing it. Both times the pattern was read as
stronger than the data supported, and both times the direction of the error favoured
the hypothesis I was hoping for. Compute the test before writing the verdict.

### What it does NOT say

**38.5% of ceiling, where the literature reports 97–99%** at comparable budgets. The
mechanism exists here but is far weaker than published, and that gap is the finding
to chase, not a footnote. Two concrete suspects in this implementation: SnapKV
selects **per head** and applies a **pooling** step so kept positions form contiguous
clusters; `attn_scores` sums over heads and selects isolated positions.

**Compute compression, not storage compression.** Scoring requires the full cache
resident, so this is not a precomputable per-bucket cartridge — a different claim
from where this track started. Stated before the run, not after.

**Not statistically established** — p = 0.219, see the significance section above.
Nothing about tonight's history of retracted headlines earns this result an exemption
just because it is the one we wanted.

Using the query to select is **not** leakage: the query is available at inference and
only the question is used, never the answer.

### Two defects found building it, both of which would have produced a wrong answer

**1. `transformers` 5.4.0 + SDPA returns `attentions=()`** — an empty tuple — and does
**not** fall back to eager. Verified directly against gpt2. Unguarded, every score
would be zero, `topk` would return positions 0..p−1, and **kv_attn would silently BE
`kv_first`** while being reported as query-aware. It would have manufactured exactly
the null result it exists to test for. `select_indices_attn` now raises and names the
fix. Loading with `attn_implementation="eager"` is mandatory for this arm.

**2. A 1-token continuation forward SIGBUSes GPT-2** on this stack. Isolated to a
minimal repro with no libucks code: `cache=199 + piece=1` dies, while 198+2, 196+4 and
150+50 all pass in separate processes — shape-specific, not the memory pressure first
suspected. `extract_bucket_kv` folds a 1-token tail into the previous segment.

### Chunked extraction, measured

CM-B.0j predicted chunking would cut the 4.5 GB permanently-cached prefill transient.
It did: `driver` fell from **11,653 MB → 9,471 MB** and the cur/driver gap from
4,651 MB → 2,683 MB. That understates the effect, because this run *also* switched to
**eager** attention, which is more memory-hungry than 0j's SDPA — so the pure chunking
win is larger than the 2.2 GB visible here, with eager eating part of it back. Memory
was flat across all 26 fixtures (`cur` 6,808 MB, `driver` 9,349–9,379 MB), confirming
0j's no-leak finding on a second code path.

### Next

~~1. Repeat this run.~~ **DONE, and it was the wrong instrument** — bit-identical, so
it proves determinism and contributes nothing to significance. Recorded because the
reasoning error is reusable: repetition tests flakiness, not effect size, and a
deterministic pipeline makes that distinction absolute.

1. **P sweep** — the highest-value next step, and it buys power two ways. Each budget
   is another paired comparison, and a *monotone* trend across P is far harder to
   obtain by chance than one point. CM-B.0i's positional selectors reached 6/26 only
   at P=1024 (4.5×); if query-aware hits the 13/26 ceiling well before that, the
   compression curve finally has a knee — which is what "compression works" looks
   like, and what 0i showed positional selection never produces.
2. **Per-head selection + SnapKV pooling** — the two named gaps to the literature.
   SnapKV selects per HEAD and pools so kept positions form contiguous clusters;
   `attn_scores` sums over heads and picks isolated positions. If these lift 5/26
   toward the published range, the effect size becomes its own evidence and the
   significance problem dissolves.
3. **A second fixture set or bucket** — the only route to more discordant pairs at
   fixed method strength, and it also tests whether any of this generalises past one
   bucket in one repo, which nothing in this track has yet established.

---

## CM-B.0j — text-in-prompt ceiling control, and the MPS memory profile (2026-07-30)

**Status**: ✅ The CM-B.0i ceiling survives its own control, and 0i reproduces exactly.

**Why this experiment.** CM-B.0i's ceiling was 13/26 — barely half, with the ENTIRE
bucket in cache. Every compression number in this log is a fraction of that number,
so if the ceiling were itself an artifact — a lossy cache path, a too-strict metric,
a too-weak model — the whole log would be measured against a bent ruler. The control
feeds the same bucket text as prompt tokens the model reads with full attention.
Causal attention makes that **mathematically identical** to a full-cache prefix, so
any gap localises a bug rather than merely suggesting one.

### Result

| arm | score | how the prefix arrived |
|---|---|---|
| floor (no memory) | **0/26** | nothing attached |
| **text in prompt** | **14/26** | live prefill, never serialised |
| **whole cache (P=5100)** | **13/26** | bf16 → CPU → float32 → `index_select` → bf16 |
| distilled cartridge | 2/26 | 98 min of distillation at P=128 |

`text − whole cache = +1`. **The ceiling is real** — model- and fixture-bound, not
harness-bound.

**The load-bearing evidence is the direction of the disagreements, not the totals.**
The two arms agree on 21/26 fixtures. Of the 5 that differ, text wins y15, y17, y22
and **cache wins y18, y24**. A lossy round-trip would lose *one-directionally*; a
symmetric ±5 scatter netting +1 is decode nondeterminism, already measured on this
hardware at ~2/25 answers. (An earlier progress note in this session claimed
"11/11, zero disagreements" at the 11-fixture mark — true then, but it did not hold
for the full run, and the weaker-but-correct argument is the directional one.)

**Consequence: the negative result stands.** `1/26 at 35.9× compression` is 8% of a
ceiling that has now been independently checked, not an artifact of a broken one.

**CM-B.0i reproduced exactly.** Whole cache 13/26 and cartridge 2/26, identical to
0i despite a different code path (the text arm routes through a new `prefix_factory`,
and `generate_answer` now raises rather than truncates) and a machine deep in swap.

### Three defects found building the control

**1. Silent truncation — third occurrence in this project.** `generate_answer`
tokenised with `truncation=True, max_length=3500`, and Qwen's `truncation_side` is
`right`. The longest control prompt is 4,626 tokens, so **1,126 tokens (24%) would
have been dropped from the TAIL** — exactly where the stratified set deliberately
places a quarter of its answers. The crippled control would have scored low, matched
the cache arm, and *confirmed* the ceiling for entirely the wrong reason. This fails
toward a **false positive**, which is worse than a crash. `generate_answer` now
raises on overflow; see `tests/unit/test_prompt_budget.py`.

**2. The distillation teacher shares that cap.** Naming the constant exposed that
both teacher paths use it, so **every cartridge distilled against this bucket was
supervised by a teacher that saw only the first 3,500 of 4,599 tokens — 76%.** On a
set where roughly a quarter of the questions target the last 24%, the cartridge could
not have learned those answers however well distillation works. Deliberately NOT
changed, because changing it silently re-scopes every prior run. But it means the
2/26 cartridge figure is **not a clean measurement of distillation capacity**:
handicapped teacher, not merely weak student. 2/26 against a floor of 0/26 is still
near-floor, so this does not overturn anything — it bounds the claim.

**3. A reuse hazard in the fix itself.** `DynamicCache` is mutated in place by the
decode loop. Handing the same object to successive fixtures would append fixture N's
question and answer to fixture N+1's prefix — silent cross-contamination of all 25
later fixtures, scores drifting upward, looking exactly like a result. The API
therefore takes a **factory, not a cache**, so freshness is structural rather than a
caller convention.

**Also verified rather than assumed:** prefilling `verbatim + "\n\n"` and forwarding
only `"Question: …\nAnswer:"` equals a single-shot prompt *only* if tokenisation is
split-invariant at the junction, and BPE is not in general. Checked token-exact on
all 26 fixtures; the script exits naming the offender if any disagrees.

### MPS memory profile — useful well beyond this experiment

| | measured over 26 fixtures |
|---|---|
| tensors (`current_allocated_memory`) | **7,002 MB, range 0 MB** |
| taken from OS (`driver_allocated_memory`) | **11,247–12,271 MB, range 1,024 MB, not climbing** |

- **No tensor leak** — `cur` pinned to the byte across the whole run.
- **No allocator leak** — `driver` plateaued after a single 610 MB warm-up fixture
  (fixture 2 added 8 MB) and then oscillated without trend.
- The **4.5 GB gap** between the two is the transient attention workspace of the
  single 4,599-token prefill, cached and **not released by
  `torch.mps.empty_cache()`** — the reading above is taken *after* an explicit call.
  With `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (mandatory here) nothing forces it back.
- **So the real requirement is 12.3 GB**, not the 7.0 GB the tensors imply. On a
  16 GB machine that is why four attempts were needed.
- **Fix worth doing regardless**: chunk long prefills into ~512-token segments —
  identical under causal attention — which should cut peak demand to ~7.5 GB.
  `extract_bucket_kv` performs the same single long forward and needs the same
  treatment. This plausibly explains MPS memory trouble in CM-A and Phase 4-C.

**Process note, recorded because it cost four launches.** After three failures I
diagnosed a per-fixture leak *by inference* and proposed chunking the run across
processes. That was wrong: two of the three failures had launched with 3–4 GB
available against a 7.2 GB requirement, and the third ran a different design. Five
lines of instrumentation settled in two fixtures what three runs of reasoning had got
backwards. **Measure allocator behaviour; never infer it.**

### Next

The ceiling is trustworthy and it is ~50%, which bounds every compression scheme on
this bucket. The mechanism this project has **never tested** is *informed* selection:
every selector in `cm_kv_prune.py` is query-agnostic — `kv_first`/`kv_last`/
`kv_stride` are positional, `kv_norm` is ‖K‖ magnitude, and its own comment concedes
it is "a cheap stand-in for attention importance." The literature does not use a
stand-in. SnapKV / H2O / [CompressKV](https://arxiv.org/html/2606.24467v1) select the
positions the **actual query** attends to and report 97–99% of full-cache accuracy at
3–19% budget; [CodeComp](https://arxiv.org/abs/2604.10235) (Apr 2026, agentic coding —
the closest prior work to libucks) selects by code-property-graph structure instead.
We measured 8% of ceiling at 3% budget with a dumb selector and concluded the
mechanism does not exist. `kv_attn` is ~30 lines against the existing
`select_indices(flat, how, p)` interface.

**Caveat to state up front**: query-aware selection needs the full cache resident at
query time to select from, so it buys **attention-compute** compression, not
**storage** compression. That is a different product claim from "small per-bucket
cartridge" — but it is the correct test of whether a compression mechanism exists
here at all, and it is the only apples-to-apples comparison with those papers.

---

## CM-B.0i — position-stratified sweep on 95c8e099 (2026-07-29)

**Status**: ✅ A clean answer, and it retracts two claims made earlier the same day.

**Why a new fixture set.** The bc6b90e2 extension set (16 fixtures, authored by
reading that bucket's text) produced an apparently clean compression curve. It was
an artifact — the scores tracked the count of fixtures whose keywords fall inside
the first P tokens almost exactly:

| P | positional ceiling | actual |
|---|---|---|
| 128 | 8/16 | 8/16 |
| 256 | 11/16 | 11/16 |
| 384 | 14/16 | 12/16 |

Those questions clustered in the head of the file, and `kv_first` keeps the FIRST P
positions. A head-clustered set cannot distinguish truncation from compression.

**The new set**: 26 fixtures for `95c8e099` (`simulation.py`, 20,013 chars / 4,599
tokens) — a different bucket and file from the one the mechanism was developed
against. Facts spread from token ~115 to ~4,818, with the spread enforced by tests
(≤35% answerable from the first eighth, ≥25% needing past the midpoint, no gap over
25% of the document). Routing deliberately bypassed via an explicit `bucket` field:
14 of the first 20 questions routed elsewhere because simulation.py is the
orchestrator, and this experiment asks whether a bucket's CACHE carries its
content, not whether the router finds the bucket.

### Result

| P | compression | score | fraction retained |
|---|---|---|---|
| floor (no memory) | — | **0/26** | — |
| 32 | 143.7× | 0/26 | 0.7% |
| 64 | 71.9× | 1/26 | 1.4% |
| 128 | 35.9× | 1/26 | 2.8% |
| 256 | 18.0× | 2/26 | 5.6% |
| 512 | 9.0× | 2/26 | 11% |
| 1024 | 4.5× | 6/26 | 22% |
| 2048 | 2.25× | 8/26 | 45% |
| **4,599 (full cache)** | **1.0×** | **13/26** | 100% |
| distilled cartridge | P=128 | 2/26 | — |

**1. Prefix truncation does not compress.** Score is proportional to fraction
retained, near-linearly. At 36× compression it sits at the positional ceiling.
There is no regime in which a small prefix carries information about text it
dropped — it answers what it literally contains.

**2. The uncompressed ceiling is 13/26 (50%).** With the entire document present as
KV, unmodified, the 3B answers half these questions and fails the other half. This
bounds every compression scheme: none can exceed the uncompressed cache. The best
compressed arm at any real ratio is 6/26 at 4.5×.

**3. Distillation ≈ truncation at matched size.** cartridge@128 2/26 vs
kv_first@128 1/26 — indistinguishable, both near zero.

### Retractions

**CM-B.0h's conclusion is withdrawn.** "98 minutes of distillation leaves the
cartridge worse than its own initialisation" (13/16 vs 8/16 at P=768) was an
artifact: `kv_first@768` covered 76% of `agents.py`, including all the early
content those fixtures asked about. It won by containing the answers, not by being
a better mechanism. At matched P on stratified fixtures the cartridge is if
anything marginally ahead.

**The CM-B.0i-precursor "compression curve" is withdrawn** for the same reason.
"3.9× retains 85%" was really "11 of my 16 questions are about the first quarter of
the file."

Both retracted claims came from fixtures written by reading the target bucket. That
is the third distinct way self-authored fixtures measured the wrong thing this
session, after the keyword-echo leak (floor 4/16) and the 8-fixture set's total
lack of resolving power (2/8 at P=32 and 2/8 at P=384).

**Two bugs in the analysis tooling, both mine:** the `min_P` helper used
`int(len/2)-1` where `grounding_score` needs `ceil(len/2)` hits — off by one for
odd keyword counts, understating head-availability; and `cm_kv_sweep` inherited
`max_chars=4096` from the distillation path, capping a 20,013-char bucket at its
first 1,021 tokens. The second was caught 45 s in from a `seq_len=1021` log line.
Recomputing the ceiling with the corrected index changes 9→8 at P=128 and 15→14 at
P=384 — the retraction stands either way.

### What survives from earlier in the day

- P=128 was genuinely too small for distillation (initial KL 5.68 → 3.63 at P=384).
- KL and grounding decouple: P=768 reached the best KL recorded here (1.093) and
  scored 2/8. Distillation overfits the 120 self-study queries.
- Neither historical "pass" reproduces — six modern draws across two configs land
  at 0–2/8 against single historical draws of 4/8 and 5/8.
- `CM_SEED` gives only approximate reproducibility; fixed-seed epoch-0 KL varied
  33% at P=384 from MPS nondeterminism.

### Where this leaves the track

**No compression mechanism has been demonstrated.** Raw cache works but does not
compress; distillation does not help. That is precisely the gap a *learned*
compressor addresses — AutoCompressor (arXiv 2305.14788) reaches 40 tokens per
summary vector by training one compressor over a corpus and encoding each document
in a single forward pass, and reports the same honest limitation we observe
independently ("summary vectors ignore some useful information accessible via full
attention"; their REPLUG top-10 text retrieval beats their own summary vectors).

The 50% uncompressed ceiling is the number to build on, because it bounds
everything else and is measured on a hostile fixture set. It argues the next
question is **model scale or an amortised compressor**, not another selection
heuristic.

**Still unknown**: whether a learned compressor works here; whether any of this
generalises past one bucket in one repo; multi-bucket composition (CAS reports
naive concatenation collapses, which is likely what CM-A.2's multi-bucket top-k
concat hit); and whether 3B is the binding limit.

---

## CM-B.0g / CM-B.0h — the P sweep, and what actually carries the content (2026-07-29)

**Status**: ❌ for context distillation, ✅ for the latent channel itself. The
memory works — the training destroys it.

### 0g — P sweep: KL improves 3×, grounding does not move

Five arms, CM-A.2's config with `CM_KEEP_BEST=1`, floor 0 on both sets in every arm.

| arm | final KL | orig 8 | ext 16 (de-leaked) |
|---|---|---|---|
| P=384 s1 | 3.317 | 1 | 5 |
| P=384 s2 | 3.859 | 4 | 9 |
| P=384 s3 | 4.034 | 1 | 9 |
| P=512 s1 | **1.151** | 4 | 7 |
| P=768 s1 | **1.093** | **2** | 8 |

P=384 across seeds: orig8 mean 2.0 (spread 3), ext16 mean 7.67 (spread 4).
Comparable back to P=128 on orig8: **0.67 → 2.25 (P≥384)**, so raising P did help.

**But P=768 has the best KL ever recorded in this project (1.093) and scores 2/8** —
worse than P=512's 4/8 at a 3× worse KL. A threefold reduction in divergence buys
nothing. This is the KL/grounding decoupling, now unambiguous.

**Diagnosis**: at high P the cartridge fits the 120 self-study queries very well;
the fixtures ask different questions. It is overfitting the self-study set, not
learning the document. Capacity was *a* constraint; removing it exposed the real
one, which is **supervision coverage**.

**Correction**: the CM-B.0b-repro finding that "200 queries vs 120 makes no
difference" was measured at P=128, where capacity was binding, so it could not
have shown a query-count effect. That test is void, not negative.

### 0h — the training-free comparison

`kv_first` is the first P positions of the **real** extracted cache. It is provably
identical to `init_from_extracted_kv` (asserted in
`test_kv_prune_selectors.py`), so `cartridge − kv_first` is exactly what the
gradient steps add.

| run | floor | **kv_first** | kv_last | kv_stride | kv_norm | **cartridge** | cart − kv_first |
|---|---|---|---|---|---|---|---|
| P=384 ext16 | 0 | **12/16** | 0 | 7 | 0 | 9/16 | **−3** |
| P=384 orig8 | 0 | 2/8 | 0 | **3** | 0 | **4/8** | +2 |
| P=768 ext16 | 0 | **13/16** | 1 | 1 | 1 | 8/16 | **−5** |
| P=768 orig8 | 0 | **3/8** | 1 | 3 | 0 | 2/8 | **−1** |

> **⚠️ RETRACTED — see CM-B.0i.** The conclusion below does not hold. These
> fixtures cluster in the first quarter of `agents.py`, and `kv_first` keeps the
> FIRST P positions, so it won by containing the answers rather than by being a
> better mechanism. On position-stratified fixtures at matched P the cartridge is
> if anything marginally ahead (2/26 vs 1/26). The numbers in the table are real;
> the interpretation was wrong.

**On 3 of 4 measurements, 98 minutes of distillation leaves the cartridge worse
than its own initialisation.** At P=768 on the 16-item set, 13/16 untrained
against 8/16 distilled.

This retro-explains the whole track: KL improves while grounding does not because
training overwrites the real activations that were already carrying the answers;
"structure, not identifiers" because the real cache *has* the identifiers and
distillation washes them out; more capacity does not help because it is more room
to overfit; and CM-A.1-retry never reproduced because it was probably never far
from the warm start.

**The positive result is the larger one**: 13/16 against a 0/16 floor, from a
single forward pass, no training. The latent channel carries repo content fine.
Compression degrades gracefully — 12/16 at P=384 (2.6×) vs 13/16 at P=768 (1.3×),
so halving the budget cost one fixture.

**Mechanism — contiguity**: `kv_first` ≫ `kv_stride` ≫ `kv_norm` ≈ `kv_last`. The
stride arm at P=768 takes 768 of 1,009 positions with irregular gaps of 1–2, which
scrambles the relative offsets RoPE encodes, and collapses to 1/16. `kv_last`
drops the file's opening, where the enum definitions most fixtures ask about live.
A **contiguous** prefix of the real cache is what works.

**Methodological gain**: the training-free arms have no seed and no optimiser, so
their only variance is MPS decode noise (~2 answers in 25). They are reproducible
in a way no cartridge in this project has been — which retires the spread-of-3-to-4
problem that made every previous number hard to read.

### Caveats

- n=1 per configuration in 0h, and orig8 disagrees at P=384 (cartridge +2). Not
  unanimous.
- The ext16 numbers are **not comparable across stages**: the de-leaked fixture
  set landed mid-sweep (01:13). Within 0g/0h they are consistent; against 0f's
  9/16 they are not.
- `CM_SEED` gives only approximate reproducibility. Two runs at identical config
  and seed produced epoch-0 KL of 3.6252 and 4.8253 — a 33% swing from MPS
  floating-point nondeterminism, far larger at P=384 than the ~1.2% seen at P=128.
  Larger P appears to buy capacity at the cost of optimisation stability.

### Consequence for the plan

**Stage 1 (Living Cartridges) is closed — and for a better reason than "it fails".**
Its premise was that rebuilding a bucket's memory costs ~7,200 s, making cheap
repair a research question. If memory is a truncated real cache, rebuild is **one
forward pass**. The problem Stage 1 was designed to solve does not exist. Do not
build the model-backed `TrialRunner`.

**Next**: CM-B.0i — sweep `kv_first` over P ∈ {32, 64, 128, 256, 384, 512, 768} on
both fixture sets to map the compression/accuracy curve. Minutes per point instead
of 98, and deterministic. This is the tradeoff curve the project has been trying to
locate since Phase 4-C.

---

## CM-B.0f — the two objections that survived the bug audit (2026-07-29)

**Status**: ⚠️ The CM-B.0e negative does not hold at larger P. **P, not the recipe,
was the binding constraint** — and every negative in this track, Phase 4-C
included, was measured at P=64 or P=128.

**Why only these two arms.** CM-B.0e found cartridge − floor ≈ 0. Most defects
found in the audit bias *downward* (verbatim truncation, unreachable fixtures,
template padding, last-epoch save), and cartridge−floor is a within-run contrast
where nearly all of them cancel: all three arms share fixtures, metric, decoder
and model. Exactly two objections survived, and neither had ever been tested.

Both arms: CM-A.2's config (120 q, 32 answer tokens, fact-probing templates),
seed 1, scored against all three floor arms on both fixture sets.

### Arm 1 — best-epoch promotion (P=128, `CM_KEEP_BEST=1`)

```
epoch 0: 5.6837   epoch 1: 3.9904  <- best, PROMOTED
epoch 2: 4.0649   epoch 3: 4.0392  <- would have shipped
```

| | floor | random | cartridge | **c−floor** |
|---|---|---|---|---|
| orig 8 | 0 | 1 | 2 | **+2** |
| ext 16 | 4 | 2 | 6 | **+2** |

Better than the +0.67 / +0.33 of three last-epoch seeds — but the KL gain was
only **0.05** (3.9904 vs 4.0392). A 1.2% KL improvement producing +2 fixtures on
both sets is not a proportionate mechanism; it is more consistent with the
fixture score being a coarse readout of an almost-unchanged cartridge. Treat as
weak.

### Arm 2 — P=384 (`CM_PREFIX_LEN=384`, last-epoch)

```
epoch 0: 3.6252   epoch 1: 3.3418
epoch 2: 3.3194   epoch 3: 4.7843  <- SHIPPED (keep_best=False on this arm)
```

| | floor | random | cartridge | **c−floor** |
|---|---|---|---|---|
| orig 8 | 0 | 1 | **3** | **+3** |
| ext 16 | 4 | 3 | **9** | **+5** |

**Initial KL fell 5.68 → 3.63 from the capacity change alone**, and this arm
shipped its *worst* epoch (4.7843 against epoch 2's 3.3194) because best-epoch
was off — so there is headroom on top of these numbers.

### Reading

P is a **positional bandwidth** constraint, not a parameter-count one. At P=128
the cartridge has 2.36 M parameters for 4,132 characters — ~570 parameters per
character, wildly overparameterised. What it lacks is *places to look*: the
verbatim occupies ~1,009 token positions and attention can only attend to P of
them. P=128 is 7.9× positional compression; P=384 is 2.6×. That is also the
cleanest explanation yet for the recurring "structure, not identifiers" failure —
structure compresses, literal strings like `source_justification` do not.

**Correction to the CM-B.0e entry**: its verdict should read "inert **at P=128**".
The conclusion was correct for the configuration measured and wrong as a general
claim about the channel. The session estimate that these two levers had a ~15–20%
chance of overturning the negative was too pessimistic.

**Still true and unchanged**: `cartridge − random` remains unquotable (a random
prefix is actively harmful, so it is a negative baseline), and the extension set
still leaks with a floor of 4/16.

**Interpretive limit for the sweep that follows.** bc6b90e2's verbatim is ~1,009
tokens, so P=128 is 7.9× compression, P=384 2.6×, P=512 2.0× and P=768 only 1.3×.
As P approaches seq_len the cartridge stops being a compression of the KV cache
and becomes a reparameterisation of it — at which point training-free KV-cache
pruning (SnapKV/H2O-style attention-score selection of the *real* cache) is the
simpler answer to the same problem, and one that cannot lose literal identifiers
because it never has to relearn them. P=768 is an upper bound, not a proposal.

**Consequence for Stage 1**: the CM-B.0e entry declared Living Cartridges dead as
designed. That is **suspended pending CM-B.0g**. If P=384 holds across seeds, a
cartridge that carries real content exists and cheap repair of it is a live
question again.

**Next**: CM-B.0g — P=384 across seeds 1–3 with best-epoch on (confirmation), then
P=512 and P=768 at one seed each (scaling).

---

## CM-B.0d / CM-B.0e — reproduction of CM-A.2, and the no-context floor (2026-07-28)

**Status**: ❌ The cartridge channel is inert on bc6b90e2. Distillation contributes
nothing measurable over having no memory at all.

### 0d — CM-A.2's exact config does not reproduce either

120 queries, `max_answer_tokens=32`, fact-probing templates, P=128, 4ep, seeds 1–3:
**0/8, 1/8, 1/8** (mean 0.67, spread 1). KL 5.754→2.987, 5.761→3.341,
6.171→4.860 — note epoch 2 *rose* in all three runs.

| run | date | queries | ans tok | qgen | score |
|---|---|---|---|---|---|
| CM-A.1-retry | Jul 2 | 200 | 48 | templates | 4/8 (n=1) |
| CM-A.2 | Jul 7 | 120 | 32 | templates | 5/8 (n=1) |
| CM-B.0b-repro | Jul 28 | 200 | 48 | templates | 2, 1, 1 |
| CM-B.0d | Jul 28 | 120 | 32 | templates | 0, 1, 1 |

**Correction to the CM-B.0b-repro entry above.** It concluded "200/48 is
reproducibly worse than 120/32". That was wrong — 120/32 measures 0.67 and 200/48
measures 1.33; both configs land in the same place. A one-fixture difference was
read as signal, immediately after warning against exactly that.

All six modern draws start at KL 5.7–6.2 with within-config spread ~0.4, against
CM-A.1-retry's claimed 0.749. Two *different* query counts landing in the same
band also refutes the earlier suggestion that query-set composition explains the
KL level. Every identifiable variable is now excluded: verbatim slicing (CM-A.2
ran after the fix), environment (torch/transformers/weights all predate Jul 2),
query count, answer budget, templates-vs-model, P, epochs, lr, and the two
scripts' setup paths. **No surviving artifact from Jul 2 or Jul 7 exists to
diagnose against.** The only two "passes" this track recorded are unreproducible
and unexplained.

### 0e — the floor, 3 arms × 3 seeds

`floor` = no prefix, `random` = untrained prefix of identical geometry,
`cartridge` = the distilled prefix. All three share the fixtures, metric, decoder
and model; `generate_answer(cartridge=None)` gives the floor, so the arms differ
in exactly one respect.

**Original 8 fixtures**

| seed | floor | random | cartridge | c−floor | c−random |
|---|---|---|---|---|---|
| 1 | 0 | 1 | 0 | +0 | −1 |
| 2 | 0 | 0 | 1 | +1 | +1 |
| 3 | 0 | 1 | 1 | +1 | +0 |
| **mean** | **0.00** | **0.67** | **0.67** | **+0.67** | **+0.00** |

**Extension 16 fixtures**

| seed | floor | random | cartridge | c−floor | c−random |
|---|---|---|---|---|---|
| 1 | 4 | 3 | 3 | −1 | +0 |
| 2 | 4 | 2 | 5 | +1 | +3 |
| 3 | 4 | 2 | 5 | +1 | +3 |
| **mean** | **4.00** | **2.33** | **4.33** | **+0.33** | **+2.00** |

**The result**: `cartridge − floor` is +0.67/8 and +0.33/16 — zero at spread ±1.
On the non-leaky set the base 3B scores **0.00/8 cold** and the cartridge, holding
the bucket's entire source distilled into 2.36M parameters, scores **0.67/8**.

### Two corrections, both to work done the same day

1. **`cartridge − random` is a misleading statistic and I introduced it.** It was
   meant to separate learned content from the mere presence of P extra attendable
   positions. But a random prefix is *actively harmful* — 2.33 vs a floor of 4.00
   on the extension set — so it is a negative baseline, not a neutral one. The
   +2.00 mean therefore mostly measures the cartridge *undoing the damage a random
   prefix does*, ending back at roughly floor level. **`cartridge − floor` is the
   only defensible statistic.** The single-seed "+3" reported earlier should not
   be quoted as a positive signal.
2. **The extension fixture set leaks.** Its floor is 4.00/16: the base model
   answers x09, x11, x12, x15 correctly with no memory, because questions like
   "what are the five states an agent can be in?" invite guesses that are right.
   Those items must be replaced before the set is used for anything.

### Consequence for the plan

**Stage 1 (Living Cartridges) is dead as designed.** Its premise is "when a bucket
changes, can its cartridge be repaired cheaply rather than rebuilt?" If the
cartridge contributes ~0 over no memory, repair is repairing nothing and the
relative claim (repair ≈ rebuild, cheaper) is trivially true and uninteresting.
Do not build the model-backed `TrialRunner`.

**Scope of the negative**: this is evidence about context distillation *at laptop
scale* — Qwen2.5-3B, P=128, ~90 min/cartridge — not about the idea. Published
Cartridges/CAS results use far larger models and orders of magnitude more compute.
It is decisive about whether the technique is available to libucks: it is not.

**Next**: CM-B.0f — the only two objections that survive the bug audit, since
most defects found bias downward and cancel in a within-run contrast:
(a) every run shipped the LAST epoch though KL rose at epoch 2 in all three 0d
seeds — `CM_KEEP_BEST=1` promotes the best; (b) P has never been varied —
`CM_PREFIX_LEN=384`. One seed each, both scored against all three floor arms. If
the cartridge still fails to beat the floor under both, the negative stands with
its plausible objections closed rather than merely unexamined.

---

## CM-B.0b-repro — CM-A.1-retry reproduction, 3 seeded draws (2026-07-28)

**Status**: ❌ The only "pass" in this track does not reproduce — and the recipe
credited for it is actively harmful.

**Ran**: bc6b90e2, the exact CM-A.1-retry recipe — 200 queries,
`max_answer_tokens=48`, fact-probing templates (`CM_MODEL_QUERIES=0`), P=128,
4 epochs, lr 1e-2, verbatim 4096, extract 1024, last-epoch save (matching the
original protocol; best-epoch selection deliberately deferred). Draw 1 unseeded,
draws 2–3 at `CM_SEED=1,2`. 12,945 s + 2×~10,000 s.

| draw | KL init → final | score |
|---|---|---|
| s0 (unseeded) | 5.763 → 4.335 | 2/8 |
| s1 (seed=1) | — | 1/8 |
| s2 (seed=2) | — | 1/8 |

**mean 1.33, range 1–2, spread 1, stdev 0.58.** First error bar this track has
ever had. The pre-registered read was "spread ≥3 means every finding collapses
into noise" — it is 1, so the findings are real.

### The recipe is the problem, not the environment

| run | date | queries | ans tok | qgen | score |
|---|---|---|---|---|---|
| CM-A.1-retry | Jul 2 | 200 | 48 | templates | 4/8 (claimed) |
| **CM-A.2** | Jul 7 | **120** | **32** | templates | **5/8** |
| CM-B.0b | Jul 27 | 120 | 32 | 0.5B model | 1/8 |
| CM-B.0b-repro | Jul 28 | 200 | 48 | templates | 1–2/8 |

Per-fixture, today's config loses **echoswarm_02, 03, 04 in all three draws** and
gains nothing. echoswarm_01 always grounds; 07/19/20 never ground in any config.
A consistent monotone loss, not sampling.

**CM-A.2 ran 2026-07-07, four days AFTER the slice-on-overflow fix (30ee434,
07-03).** It therefore had the identical post-fix verbatim, identical templates,
identical P and epochs. The only differences from today are query count
(120 vs 200) and answer budget (32 vs 48). So:

- **Verbatim slicing is ruled out** as the cause. `CM_VERBATIM_CHARS` still
  reproduces the pre-fix text byte-for-byte (=3868 → 3898 chars, clean chunk
  boundary) and remains available, but it is no longer the suspect.
- **Environment drift is ruled out**: torch 2.11.0 (site-packages May 14),
  transformers 5.4.0 (Mar 31), safetensors (Mar 31), Qwen2.5-3B weights snapshot
  `3aab1f19` (Apr 4). All predate Jul 2. The only lockfile change since
  (`4d6d982`) merely dropped `bitsandbytes` on macOS.
- **CM-B.0b's collapse was not mainly the 0.5B generator.** Templates at 200/48
  also give 1/8. Two independent changes were each harmful.

### Corrections to the record

1. **KL is not comparable across runs with different query sets.** It is measured
   against teacher answers for that run's queries, and the query sets differ (the
   identifier list shifts: `agent`/`bool` in, `n_drop`/`pop` out). An earlier
   claim in this session that "the 20× KL gap is too stable to be noise" was
   unsound. The score comparison, called the weaker evidence at the time, is what
   held up.
2. **The eval is not bit-deterministic.** `echoswarm_cartridge_A2.json` and
   `A2_r1_fail7.json` are the same cartridge scored twice: verdicts identical
   (7/25), but answer TEXT differs on echoswarm_10 and echoswarm_16. CM-A.2's
   "bit-identical across 2 evals" is wrong — the score matched, the text did not.
   Likely MPS reduction ordering flipping an argmax.
3. **`_collect_source_text` overshoots `max_chars`** by (blocks−1)×6, because it
   sums block lengths but joins with a 6-char separator (+36 at the 4096
   default). Left unchanged and documented: fixing it would shift every verbatim
   length and break comparability. Also: pre-fix verbatim is 3898 chars, not the
   3868 quoted earlier in this session (that figure omitted separators).

### Leading hypothesis

At P=128 the prefix cannot absorb 200 queries' worth of signal, and 48-token
teacher answers dilute it further — so *more* self-study makes the cartridge
worse, not better. This is the opposite of the CM-A.2 log's own recommendation to
scale queries to 512–1000, which should now be considered withdrawn.

**Next**: CM-B.0d — reproduce CM-A.2's exact config (120 q, 32 answer tokens,
templates) on bc6b90e2 across 3 seeds. If 5/8 returns, the finding is "more
queries and longer answers hurt at P=128" and we have a working recipe for the
first time. If it does not, CM-A.2's 5/8 was itself one lucky draw and nothing in
this track has ever worked — which supersedes Stage 1 and every downstream plan.
Also queued: the 16-fixture bc6b90e2 extension set (24 items total) for a
better-powered read, scored separately since it changes the denominator.

---

## CM-B.0b — Query-gen fix + 3-bucket re-distill (2026-07-27)

**Status**: ❌ GATE FAIL — and a regression against the bucket it was meant to help.

**Ran**: `cm_distill_buckets.py --buckets bc6b90e2,40615ba9,fe7ded0d --force`,
3h32m + 2 buckets, 0 failures. Query generator wired per plan (`Qwen2.5-0.5B-Instruct`).

| bucket | KL start → end | time | grounding before → after |
|---|---|---|---|
| `40615ba9` | 0.536 → **0.227** | 8,134 s | 0/3 → 0/3 |
| `bc6b90e2` | 5.443 → **5.228** | 6,796 s | 5/8 → **1/8** |
| `fe7ded0d` | 3.880 → **2.749** | 5,874 s | 0/2 → 0/2 |

**Gate**: the plan said "currently 2/13, pass at ≥5/13". That 2/13 was scored with the
**pre-0a** metric. Re-scoring the *unchanged* CM-A.2 cartridges with `grounding_score`
gives **5/13** (bc6b90e2 5/8, 40615ba9 0/3, fe7ded0d 0/2). So ≥5/13 was the *baseline*,
not a bar. Result **1/13**. The gate text in `cm-b-plan.md` has been corrected.

**Six of thirteen answers are degenerate token loops** — `'TheORYoryoryory…'`,
`'A Skeptical agent isatisatisatis…'` — all from bc6b90e2. Not weak grounding; the
cartridge is corrupting generation.

### The query-gen hypothesis is dead

`fe7ded0d` was chosen to isolate it: 6 clean re-distills under the old template config
gave KL 3.69→2.69; fact-probing queries gave **3.880→2.749**. Same trajectory,
marginally worse. The generator was never the lever.

It also only partly applied: the 0.5B model produced **84/120** distinct questions for
bc6b90e2 and **81/120** for fe7ded0d — the shortfall was silently padded with the exact
templates CM-A.1 identified as the bottleneck. `40615ba9` got 112/120 and had the best
KL by 20×. The `if not qs` guard at `:186` can never fire, because
`generate_self_study_queries` always tops up to `n`; it was giving false assurance.

### KL is decoupled from grounding — proven both directions

`40615ba9` reached **KL 0.227** — better than CM-A.1-retry's passing 0.219 — and
grounded **0/3**. Do not read KL as a proxy for the gate. CM-A.2 made the same mistake.

### Diagnostics (zero compute)

**1. The 4096-char verbatim cap discards most of some buckets.**

| bucket | true content | distilled | discarded |
|---|---|---|---|
| `40615ba9` | 14,852 ch | 4,126 | **72%** |
| `fe7ded0d` | 11,999 ch | 4,102 | **66%** |
| `bc6b90e2` | 4,661 ch | 4,132 | 11% |

Query-gen sees only `bucket_text[:3000]` (`self_study.py:98`) — 73% of the kept text,
20% of 40615ba9's real content. Cross-bucket KL comparisons are invalid while
truncation ratios differ this much: a heavily-truncated bucket has an easier target.

**2. The eval ceiling is 20/25, not 25/25.** Under `grounding_score` (≥50% of keywords),
five fixtures cannot be answered from the routed bucket's kept verbatim:
echoswarm_05 (1/3), 06 (0/3), 07 (0/4), 16 (0/5), 23 (1/5). On the 13-fixture subset the
ceiling is **10/13**. Raising the cap would fix only **05 and 06** — 07, 16 and 23 fail
because their keywords are not in the routed bucket at all, which is a *routing* defect.

**3. Truncation is NOT what blocks 40615ba9.** All six identifiers echoswarm_11 needs
(`who`, `what`, `where`, `when`, `which_route`, `source_justification`) were inside the
3,000 chars query-gen read *and* the 4,126 distilled. KL 0.227. The answer invented
`"message"`, `"location"`, `"time"`… — fluent, plausible, wrong. Nothing was missing,
nothing unqueried, convergence excellent, identifiers still not retained. This is
CM-A.2's "structure, not identifiers" under the cleanest possible conditions.

**4. The truncation fix (`30ee434`, 2026-07-03) is not the regression cause.** Pre- vs
post-fix verbatim: 40615ba9 3,325→4,126, bc6b90e2 **3,868→4,132 (+7%)**, fe7ded0d
2,809→4,102. A 7% content change cannot explain bc6b90e2's initial KL going 0.749→5.443.

**The one variable never tested**: CM-A.1-retry used **200 queries and
`max_answer_tokens=48`**. CM-A.2 used 120/32. CM-B.0b used **120/32 again**. Listing six
JSON keys does not fit in 32 tokens, so the teacher never demonstrated the full fact.

**Next**: CM-B.0c — faithful CM-A.1-retry reproduction on bc6b90e2 (200 q,
`max_answer_tokens=48`, P=128, 4ep, last-epoch save to match the original protocol).
8 fixtures = the best-powered readout available. If 4/8 does not return, no result in
this track is trustworthy and that supersedes every downstream plan.

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
