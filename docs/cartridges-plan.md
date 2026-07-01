# Plan — Cartridge Memory (CM): self-distilled latent memory, fresh track

> Fresh start, not "Phase 5." Prior work (V1, V2, Phase 3/4-A/4-C) is treated as
> history to be archived. New track codename **Cartridge Memory (CM)**, stages
> **CM-0 → CM-A → CM-B → CM-C**. Rename freely — nothing depends on the codename.

## Context

Phase 4-C closed as a **negative result**: libucks's latent channel is inert. The
fairness eval (`docs/phase-4c-log.md` 4-C.6-FAIRNESS) showed `cache_aug_no_verbatim`
= 2/25, *below* the no-context floor (3/25) — all grounding comes from the verbatim
(RAG) channel. Two implementations (adapter soft-prompt, DeepMind-style coprocessor)
failed the same way.

Root cause, confirmed by codebase map + 2025–2026 literature:

1. **Wrong training objective.** libucks trains the latent with cosine-to-token-
   embeddings + margin/JSD separation (`_cli.py:1442–1468`, `losses.py:11–78`). No
   context distillation. Methods that make a latent channel *carry facts* use
   **context distillation** — train the latent so the frozen model, given only the
   latent, reproduces the logits it produces *with full text in context*.
   - "Context Distillation as Latent Memory Management" (arXiv 2605.28889, 2026):
     KL(teacher(y|q,context) ‖ student(y|latent)). SQuAD **36.3 EM vs ~0–1**;
     NarrativeQA **28.6 ROUGE vs ~4**. Qwen2.5-7B.
   - Cartridges / self-study (arXiv 2506.06266, Stanford): self-generate synthetic
     conversations about a corpus → context-distill into a trainable **KV-prefix**
     (frozen model). Matches in-context learning at **38.6× less memory**.
2. **Toy-scale data.** `qa_per_bucket` defaults to **1** (`data_generator.py`,
   `_cli.py:580–636`); working papers use **~1000 synthetic queries per document**.
3. **No per-bucket latent artifact / no self-injection.** `adapter.pt`/`lora_receiver.pt`
   are global; on create/update/split/merge only centroid+prose refresh
   (`librarian.py:149–200`, `mitosis.py`, `merging_service.py`). That absence is where
   "self-injection" belongs — and the *refresh-on-change* loop is the unclaimed novelty
   (2605.28889 and Cartridges both leave updates unaddressed).

**Innovation honesty:** CM-A is a deliberate *replication* (Cartridges at your scale) to
de-risk cheaply — not itself novel. The novel end-state is CM-B (self-refresh loop on live
code) + CM-C (cross-repo cartridge fusion), contingent on CM-A proving the latent channel
can carry facts. Decisions locked with user: minimal-proof-first; MPS/frozen Qwen2.5-3B/$0;
trainable KV-prefix cartridge.

---

## CM-0 — Documentation cleanup & archive (do first)

Goal: shrink the doc surface to what's current before adding new plan/log files, so the
new track starts on a clean tree. Create `docs/archive/` (git-tracked) and move superseded
docs there — **archive, don't delete** (history is preserved; `git mv` keeps blame).

**Proposed classification** (root `.md` + `docs/`). Items marked *review* need a quick read
during execution to confirm before moving:

| file | action | rationale |
|---|---|---|
| `CLAUDE.md`, `README.md`, `ARCHITECTURE.md`, `POCSTRAT.md` | **KEEP** | active blueprints / strategy |
| `QUICKSTART.md` | **ARCHIVE** | V1 quickstart, superseded by `QUICKSTART_V2.md` |
| `QUICKSTART_V2.md` | KEEP (rename → `QUICKSTART.md` after old one archived) | current quickstart |
| `IMPLEMENTATION_PLAN.md` | **ARCHIVE** + repoint CLAUDE.md | V1 plan; superseded |
| `V2_IMPLEMENTATION_PLAN.md` | **ARCHIVE** | V2 plan; superseded by phase-4c + CM |
| `INTERLAT_LITE_MANUAL.md` | *review* → archive if stale | likely a Phase-2 how-to; keep only if still accurate |
| `LATENT_RAG_ARCHITECTURE.md` | *review* → fold into `ARCHITECTURE.md` or archive | probably subsumed |
| `PITCH.md` | *review* → keep or fold into `POCSTRAT.md` | positioning duplication |
| `docs/phase-4c-plan.md`, `docs/phase-4c-log.md` | **ARCHIVE** → `docs/archive/phase-4c/` | closed phase; keep as history |
| `docs/*.md` (5 arXiv) + `docs/new-docs/*` (10 arXiv) | **CONSOLIDATE** → `docs/papers/` | reference, not outdated; dedupe the `2412.06769` copy; not archived |

**CLAUDE.md pointer fixes (required by the archive moves):**
- §1 "Project Blueprints" references `IMPLEMENTATION_PLAN.md` → repoint to
  `docs/cartridges-plan.md` (+ note V1/V2 plans are in `docs/archive/`).
- §3 "Session Start Protocol" step 1 references `IMPLEMENTATION_PLAN.md` → repoint to
  `docs/cartridges-log.md`.
- Leave the golden rules (mitosis/novel-bucket/query-dropout) intact; they still hold.

**Gate:** `git status` clean of stray docs; `rg -l 'IMPLEMENTATION_PLAN\.md|QUICKSTART_V2'`
returns no live references except inside `docs/archive/`; repo root has ≤6 `.md` files.
Commit as a standalone `docs: archive superseded plans, consolidate papers` commit.

---

## CM-A — Hypothesis proof (context-distilled KV cartridge, 1 repo, MPS/3B, $0)

**H:** a context-distilled per-bucket trainable KV-prefix cartridge lets the frozen 3B
answer bucket questions from the **latent alone** (no verbatim).

**Gate (echoswarm, 25 fixtures):**
- Primary: `cartridge` (latent-alone) grounding **≥ 8/25** (= no_context 3 + 5, the
  "strong win" bar), up from cache_aug_no_verbatim's 2/25.
- Secondary: `cartridge` cosine ≫ 0.549 (the inert baseline), into hybrid range.
- Sanity: distillation KL to the full-context teacher decreases; grads flow **only** into
  the cartridge prefix (frozen-model assertion).
- Met → CM-B/CM-C. Not met → documented second negative result; 3B is the ceiling.

### Design (Cartridges recipe)
- **Artifact:** per-bucket trainable KV-prefix — learnable `(K,V)` per layer,
  `(n_layers=36, n_kv_heads=2, P=64, head_dim=128)`, prefix-tuning on frozen Qwen2.5-3B.
  **Init from the bucket's real extracted KV** (`kv_extract.extract_bucket_kv`, on disk via
  `BucketKVCache`). Store `.libucks/kv_cache/<bucket_id>.cartridge.safetensors`.
- **Self-study (local, $0):** per bucket, generate ~128 diverse synthetic queries about its
  chunks (scale toward ~512–1000 only if the gate is borderline). Teacher target = the
  frozen 3B's own output with full verbatim in context; no gold labels, no API needed.
- **Objective — context distillation:** per (bucket, query q): teacher forward
  `[verbatim, q]`→logits_teacher; student forward `[cartridge_prefix, q]`→logits_student;
  **loss = KL(softmax(logits_teacher/T) ‖ softmax(logits_student/T))** over teacher-greedy
  continuation positions (+ optional CE on greedy tokens). Backprop **only** into the prefix.
- **Decode/eval:** load cartridge → supply as `past_key_values` prefix (reuse `decode.py`
  concat, swapping `coprocessor(z)` for the trained cartridge). Latent-alone = no verbatim.
  Multi-bucket in CM-A = concat top-k cartridge prefixes (no fusion training yet).

### Sequencing (TDD, phase-gated per CLAUDE.md)
- **CM-A.0** — scaffold `docs/cartridges-plan.md` + `docs/cartridges-log.md`; write smoke
  test first (watch it fail).
- **CM-A.1 — single-bucket proof:** distill ONE echoswarm bucket; show latent-alone answers
  ≥3 held-out questions about *that* bucket (vs current inert). Fastest falsification.
- **CM-A.2 — all-bucket + eval gate:** distill all 33 buckets; run eval; check the gate.

## Scope boundaries (NOT in CM-A)
- **CM-B (contingent):** self-refresh loop — re-distill a bucket's cartridge on
  create/update/split/merge (`Librarian._handle_update`, `mitosis.py`, `merging_service.py`,
  `NovelBucketService`). The novel "living memory" contribution.
- **CM-C (contingent, may need cloud):** shared cross-repo layer — trained fusion /
  entropy-gated router (2605.28889) or shared KV routers (2508.17032); per-bucket cartridges
  stay local. Revisit cloud only here.
- No changes to routing, embeddings, MCP, or the Phase 4-A hybrid production path.

## Critical files

### NEW
- `libucks/cache_augmentation/cartridge.py` — `KVPrefixCartridge(nn.Module)`: per-layer
  trainable K,V; `init_from_extracted_kv`; `save`/`load`; yields `past_key_values`.
- `libucks/thinking/training/cartridge_trainer.py` — context-distillation loop (teacher =
  full-verbatim frozen 3B; student = cartridge; KL loss; backprop into prefix only).
- `libucks/thinking/training/self_study.py` — scaled per-bucket synthetic query generation.
- `tests/integration/test_cartridge_smoke.py`, `tests/unit/test_cartridge_data.py`.

### MODIFIED
- `libucks/thinking/training/losses.py` — add `distillation_loss` (KL to full-context
  teacher, temperature), styled after `losses.py:81–103`.
- `libucks/cache_augmentation/decode.py` — accept a trained-cartridge prefix; top-k concat.
- `libucks/cache_augmentation/bucket_kv_cache.py` — store/load `.cartridge` variant; reuse
  the `(chunk_id, git_sha)` staleness signature.
- `libucks/_cli.py` — `libucks distill-cartridges --repo X --queries-per-bucket N`; raise
  the query-per-bucket ceiling.
- `tests/eval/test_latent_vs_baseline.py` — add `cartridge` (latent-alone) + `cartridge_hybrid`.

### REUSED UNCHANGED
- `kv_extract.py` (cartridge init); `coprocessor.py`/`fusion.py` (untouched in CM-A);
  `health_monitor.py`/`mitosis.py`/`merging_service.py` (CM-B); routing/embeddings/MCP.

## Verification (end-to-end)
1. `uv run pytest tests/integration/test_cartridge_smoke.py` — one distill step: KL emitted,
   prefix `.grad` non-None, all base-model params `.grad is None` (frozen check).
2. CM-A.1: distill one bucket (~128 queries); held-out KL ≪ init KL; ≥3 fixtures answered
   from latent alone.
3. CM-A.2: `LIBUCKS_EVAL_REPOS=echoswarm uv run pytest -m eval tests/eval/test_latent_vs_baseline.py -v -s`
   → `cartridge` grounding ≥ 8/25, cos ≫ 0.549. Snapshot to
   `tests/eval/results/cm/echoswarm_cartridge_A2.json`.
4. Record numbers + verdict in `docs/cartridges-log.md`; add memory `project_cartridge_result`.

## Compute reality (CM-A, MPS/3B)
33 buckets × ~128 self-study queries ≈ 4.2k teacher+student forwards + prefix backward;
per-bucket cartridges train independently (sequential). CM-A.1 single bucket ~minutes–1h to
validate the pipeline; CM-A.2 batch overnight. Start 128/bucket; scale only if borderline.
Zero API/cloud cost.
