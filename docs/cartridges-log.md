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
- CM-B.0e — no-context floor, 3 arms × 3 seeds — ❌ **LATENT CHANNEL INERT** 2026-07-28 (**cartridge − floor = +0.67/8 and +0.33/16, i.e. zero at spread ±1. On the non-leaky set the base 3B scores 0.00/8 cold and the cartridge scores 0.67/8. A random prefix is ACTIVELY HARMFUL (2.33 vs floor 4.00), so `cartridge − random` overstates the benefit and must not be quoted**)

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
