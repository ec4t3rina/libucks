"""Phase 1A eval harness — latent vs text-baselines vs no-context.

Run via:
    pytest -m eval tests/eval/test_latent_vs_baseline.py -v -s

What it measures (per hand-curated fixture):
  - Routing accuracy   — does the top-routed bucket text contain the expected
                         keywords? (latent + text paths share routing → tied)
  - Answer grounding   — does the generated answer contain >= 50% of
                         answer_keywords?
  - Cosine similarity  — sentence-transformer cosine between answer and the
                         ground-truth answer.

Four generation paths per fixture:
  1. latent      — current pipeline: top-k routing → Librarian Representations
                   → CommunicationAdapter (inter-Librarian self-attention) →
                   Translator decode (with LoRA).
  2. text_lora   — same top-k routing, but the 3 bucket *proses* are fed as a
                   text prompt to the base receiver with LoRA active.
  3. text_clean  — same as text_lora but with LoRA scaling temporarily set to
                   0.0 so lora_delta = 0 (= base model only).
  4. no_context  — base model alone, no bucket context. Sanity floor.

We report all four so we can attribute any gain: if text_lora >> text_clean,
LoRA is helping text inputs (= specialization leak); if latent > text_lora,
the latent path is communicating more than the bucket text alone carries.

This harness is not collected by default (pytest mark 'eval'). It depends on
the target repo's .libucks/ being initialised with adapter.pt + lora_receiver.pt
+ w_a.pt + a populated registry.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pytest
import torch

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.librarian import Librarian, _collect_source_text
from libucks.query_orchestrator import QueryOrchestrator
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking import create_strategy
from libucks.thinking.communication_adapter import CommunicationAdapter
from libucks.thinking.training.lora_trainer import LoRALinear, _inject_lora, _LORA_TARGETS
from libucks.translator import Translator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target repos and their fixture files. libugry is the primary Phase 1
# benchmark — it's the user's own brand-new repo, so Qwen has never seen it
# and `no_context` should be near zero. Click stays as a sanity check (heavy
# Qwen contamination → useful only for "did retraining regress?").
_REPOS: list[Tuple[str, Path, Path]] = [
    (
        "libugry",
        Path("/Users/ecaterina/Developer/test-repos/libugry"),
        Path(__file__).parent / "fixtures" / "libugry_qa.json",
    ),
    (
        "click",
        Path("/Users/ecaterina/Developer/test-repos/click/src/click"),
        Path(__file__).parent / "fixtures" / "click_qa.json",
    ),
    (
        "echoswarm",
        Path("/Users/ecaterina/Developer/test-repos/echoswarm"),
        Path(__file__).parent / "fixtures" / "echoswarm_qa.json",
    ),
]

# Optional comma-separated repo filter (e.g. LIBUCKS_EVAL_REPOS=libugry) so a
# phase that only cares about one repo can skip the others. No-op when unset.
_repo_filter = os.environ.get("LIBUCKS_EVAL_REPOS")
if _repo_filter:
    _wanted = {r.strip() for r in _repo_filter.split(",") if r.strip()}
    _REPOS = [r for r in _REPOS if r[0] in _wanted]

# Max new tokens per generation. Click answers are short; keep cheap.
_MAX_NEW_TOKENS = 120

# Token cap on per-bucket prose injected into text prompts. Keeps the prompt
# tractable on a 4096-token-cap MPS receiver.
_BUCKET_PROSE_CAP_CHARS = 1200


# ---------------------------------------------------------------------------
# Pipeline assembly (mirrors mcp_bridge._load_heavy but synchronous, eval-only)
# ---------------------------------------------------------------------------

def _build_pipeline(repo_path: Path):
    """Load registry, store, strategy + LoRA, adapter, translator for a repo."""
    cfg = Config.load(repo_path)
    libucks_dir = repo_path / ".libucks"

    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    store = BucketStore(libucks_dir / "buckets")

    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    strategy = create_strategy(cfg)

    strategy._mgr.load_base_model(
        model_id=cfg.model.base_model,
        quantization=cfg.model.quantization,
        bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
        device=cfg.model.device,
        base_model_dtype=cfg.model.base_model_dtype,
    )
    base_model = strategy._mgr.get_base_model()
    base_tokenizer = strategy._mgr.get_base_tokenizer()

    # Inject LoRA + load weights (match mcp_bridge.py:130). r=16/alpha=16.0
    # matches the lsep_v3 trained checkpoint.
    lora_path = libucks_dir / "lora_receiver.pt"
    if lora_path.exists():
        from libucks.thinking.model_manager import ModelManager as _MM
        _resolved = _MM._resolve_device(cfg.model.device)
        _inject_lora(base_model, _LORA_TARGETS, r=16, alpha=16.0)
        _lora_state = torch.load(lora_path, map_location=_resolved, weights_only=True)
        base_model.load_state_dict(_lora_state, strict=False)

    # CommunicationAdapter for the latent path.
    from transformers import AutoConfig as _AC
    _enc_dim = _AC.from_pretrained(cfg.model.local_model).hidden_size
    _base_dim = _AC.from_pretrained(cfg.model.base_model).hidden_size
    adapter = CommunicationAdapter(hidden_dim=_enc_dim, output_dim=_base_dim,
                                   output_len=cfg.model.output_len)
    adapter_path = libucks_dir / "adapter.pt"
    if adapter_path.exists():
        adapter.load_saved_weights(adapter_path)
    _adapter_dtype = base_model.dtype
    from libucks.thinking.model_manager import ModelManager as _MM
    _adapter_device = _MM._resolve_device(cfg.model.device)
    adapter = adapter.to(device=_adapter_device, dtype=_adapter_dtype)

    # Phase 3-B: per-chunk embedding retriever powers both routing rerank
    # and verbatim chunk selection. Cache lives on disk for repeated runs.
    from libucks.chunk_retriever import ChunkRetriever
    chunk_retriever = ChunkRetriever(
        cache_dir=libucks_dir / "chunk_emb_cache",
        embedder=embedder,
        store=store,
    )

    translator = Translator(strategy, adapter=adapter, chunk_retriever=chunk_retriever)
    # Hybrid translator: same adapter + strategy, but with the verbatim
    # retrieval channel enabled. Used by _path_hybrid for direct comparison.
    hybrid_translator = Translator(
        strategy, adapter=adapter, store=store, hybrid=True,
        verbatim_max_chars=cfg.model.hybrid_verbatim_max_chars,
        chunk_retriever=chunk_retriever,
    )

    # Phase 4-C cache_aug translator (lazy — only built if state_dict + KV cache
    # are present on disk). Loads Qwen 2.5-3B as a SEPARATE frozen receiver
    # (the Phase 4-A receiver in pipe["base_model"] may be a smaller model).
    cache_aug_state_path = libucks_dir / "cache_aug_state.pt"
    cache_aug_kv_dir = libucks_dir / "kv_cache"
    cache_aug_translator = None
    cache_aug_receiver = None
    if cache_aug_state_path.exists() and cache_aug_kv_dir.is_dir():
        from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
        from libucks.cache_augmentation.coprocessor import Coprocessor
        from libucks.cache_augmentation.fusion import CrossBucketFusion
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from libucks.thinking.model_manager import ModelManager as _MM
        _cache_aug_device = _MM._resolve_device(cfg.model.device)
        _cache_aug_receiver_id = "Qwen/Qwen2.5-3B"
        cache_aug_receiver = AutoModelForCausalLM.from_pretrained(
            _cache_aug_receiver_id, dtype=torch.bfloat16,
        ).eval().to(_cache_aug_device)
        cache_aug_tokenizer = AutoTokenizer.from_pretrained(_cache_aug_receiver_id)
        _coproc = Coprocessor().to(_cache_aug_device).to(torch.float32)
        _fusion = CrossBucketFusion().to(_cache_aug_device).to(torch.float32)
        _state = torch.load(cache_aug_state_path, map_location=_cache_aug_device, weights_only=True)
        _coproc.load_state_dict(_state["coproc"])
        _fusion.load_state_dict(_state["fusion"])
        _coproc.eval(); _fusion.eval()
        _kv_cache_obj = BucketKVCache(cache_aug_kv_dir, model_id=_cache_aug_receiver_id)
        _bucket_chunks = {bid: list(store.read(bid)[0].chunks) for bid in registry.get_all_centroids()}
        cache_aug_translator = Translator(
            strategy, adapter=None, store=store, hybrid=True,
            verbatim_max_chars=cfg.model.hybrid_verbatim_max_chars,
            chunk_retriever=chunk_retriever,
            cache_aug={
                "coproc": _coproc, "fusion": _fusion,
                "kv_cache": _kv_cache_obj, "receiver": cache_aug_receiver,
                "tokenizer": cache_aug_tokenizer, "device": _cache_aug_device,
                "bucket_chunks": _bucket_chunks,
            },
        )

    librarians: dict[str, Librarian] = {}
    for bid in registry.get_all_centroids():
        lib = Librarian(
            bucket_id=bid,
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
            repo_path=repo_path,
            translator=translator,
            chunk_retriever=chunk_retriever,
        )
        librarians[bid] = lib

    # Phase 4-A ablation: LIBUCKS_DISABLE_ROUTING_RERANK=1 turns off the
    # routing-side chunk rerank (centroid-only routing) while keeping the
    # in-bucket chunk rerank used by the Translator's verbatim selection.
    # Isolates which lever of Phase 3-B's chunk rerank did the work.
    _routing_retriever = None if os.environ.get("LIBUCKS_DISABLE_ROUTING_RERANK") else chunk_retriever
    agent = CentralAgent(
        registry, cfg, embed_fn=embedder.embed,
        chunk_retriever=_routing_retriever,
    )
    for bid, lib in librarians.items():
        agent.register_librarian(bid, lib)

    orchestrator = QueryOrchestrator(
        central_agent=agent,
        librarians=librarians,
        embed_fn=embedder.embed,
        top_k=cfg.routing.top_k,
    )

    return {
        "cfg": cfg,
        "registry": registry,
        "store": store,
        "embedder": embedder,
        "strategy": strategy,
        "base_model": base_model,
        "base_tokenizer": base_tokenizer,
        "adapter": adapter,
        "translator": translator,
        "hybrid_translator": hybrid_translator,
        "cache_aug_translator": cache_aug_translator,
        "orchestrator": orchestrator,
        "agent": agent,
        "librarians": librarians,
    }


@contextmanager
def _lora_disabled(model) -> Iterator[None]:
    """Temporarily zero every LoRALinear's scaling so lora_delta = 0.

    Equivalent to running the base model without LoRA. Restores the original
    scaling on exit. Cheap — touches a Python float per module, no tensors.
    """
    saved: list[Tuple[LoRALinear, float]] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            saved.append((module, module.lora_scaling))
            module.lora_scaling = 0.0
    try:
        yield
    finally:
        for module, scaling in saved:
            module.lora_scaling = scaling


# ---------------------------------------------------------------------------
# Generation paths
# ---------------------------------------------------------------------------

def _generate_text(model, tokenizer, prompt: str, device) -> str:
    """Tokenize prompt, generate up to _MAX_NEW_TOKENS, decode the continuation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the prompt tokens from the output.
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _build_text_prompt(question: str, bucket_proses: List[str]) -> str:
    """Plain RAG-style prompt: context blocks + question."""
    ctx_blocks = []
    for i, prose in enumerate(bucket_proses, start=1):
        snippet = prose.strip()[:_BUCKET_PROSE_CAP_CHARS]
        ctx_blocks.append(f"[Context {i}]\n{snippet}")
    ctx = "\n\n".join(ctx_blocks)
    return (
        f"{ctx}\n\n"
        f"Based on the context above, answer the following question concisely "
        f"(2-3 sentences, technical).\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _build_no_context_prompt(question: str) -> str:
    """No-context prompt: deliberately repo-agnostic so the same template works
    across click and libugry. Telling the model 'this is about <library X>'
    would either help (click) or mislead (libugry has no prior); generic is
    the cleanest sanity floor."""
    return (
        f"Answer the following technical question about a Python codebase "
        f"concisely (2-3 sentences).\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


async def _path_latent(pipe, question: str) -> Tuple[str, List[str], list, np.ndarray]:
    """Top-k routing → Librarian Representations → adapter → Translator decode.

    Returns (answer, bucket_ids, reps, query_embedding). The embedding is
    returned so downstream paths (hybrid, text_*) can pass it into
    synthesize / chunk-rerank without re-embedding."""
    query_embedding = pipe["embedder"].embed(question)
    pairs = await pipe["orchestrator"].query(question)
    reps = [rep for _, rep in pairs]
    bucket_ids = pipe["agent"].route(query_embedding, pipe["cfg"].routing.top_k)
    answer = await pipe["translator"].synthesize(
        question, reps, bucket_ids=bucket_ids, query_embedding=query_embedding,
    )
    return answer, bucket_ids, reps, query_embedding


async def _path_hybrid(
    pipe, question: str, bucket_ids: List[str], reps: list,
    query_embedding: np.ndarray,
) -> str:
    """Latent + verbatim channel + LoRA active. Reuses the reps from
    _path_latent so single-question cost is one extra decode."""
    return await pipe["hybrid_translator"].synthesize(
        question, reps, bucket_ids=bucket_ids, query_embedding=query_embedding,
    )


def _path_text(pipe, question: str, bucket_ids: List[str], lora_active: bool) -> str:
    """Read the bucket source code (same content the latent path sees via
    Librarian._handle_query → _collect_source_text), build a text prompt,
    and generate. Reading `prose` would be wrong here because broken-decoder
    repos like click have empty prose — that would silently degrade the
    text-baseline to no_context-with-whitespace."""
    bucket_contents: List[str] = []
    for bid in bucket_ids:
        try:
            front_matter, prose = pipe["store"].read(bid)
        except FileNotFoundError:
            continue
        # Match the latent path's behaviour: prefer real source over (possibly
        # empty / hallucinated) prose. max_chars matches librarian.py:310.
        content = _collect_source_text(front_matter, max_chars=_BUCKET_PROSE_CAP_CHARS) or prose
        bucket_contents.append(content)
    prompt = _build_text_prompt(question, bucket_contents)
    device = pipe["base_model"].device
    if lora_active:
        return _generate_text(pipe["base_model"], pipe["base_tokenizer"], prompt, device)
    with _lora_disabled(pipe["base_model"]):
        return _generate_text(pipe["base_model"], pipe["base_tokenizer"], prompt, device)


def _path_no_context(pipe, question: str) -> str:
    prompt = _build_no_context_prompt(question)
    device = pipe["base_model"].device
    with _lora_disabled(pipe["base_model"]):
        return _generate_text(pipe["base_model"], pipe["base_tokenizer"], prompt, device)


def _path_cache_aug(
    pipe, question: str, bucket_ids: List[str], query_embedding: np.ndarray,
    *, use_verbatim: bool = True, cold_stop_entropy: Optional[float] = 4.0,
) -> str:
    """Phase 4-C cache-augmentation path. Returns "" if cache_aug isn't loaded
    (so the row is excluded from grounding when the feature isn't enabled
    for this repo).

    Phase 4-C.6 fairness variants (same 3B receiver bundle, no extra load):
      use_verbatim=False     → cache_aug_no_verbatim (latent-alone gate).
      cold_stop_entropy=None → cache_aug_greedy_nogate (attribute the win
                               to Cold Stop vs. plain greedy)."""
    translator = pipe.get("cache_aug_translator")
    if translator is None:
        return ""
    return translator.synthesize_cache_aug(
        question, bucket_ids, query_embedding=query_embedding,
        use_verbatim=use_verbatim, cold_stop_entropy=cold_stop_entropy,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _routing_score(
    pipe,
    bucket_ids: List[str],
    expected_keywords: List[str],
    question: str,
) -> bool:
    """Routing accuracy: True if EITHER any expected keyword appears in the
    combined bucket text OR any routed bucket's centroid is cos > 0.5 from the
    query embedding. The previous majority-keyword check was 14/15 false-negative
    on libugry because HealthMonitor regenerates prose to a summary form that
    loses literal keywords even when the bucket's source code clearly answers
    the question."""
    combined = []
    for bid in bucket_ids:
        try:
            front_matter, prose = pipe["store"].read(bid)
        except FileNotFoundError:
            continue
        combined.append(prose)
        combined.append(front_matter.domain_label or "")
        for chunk in front_matter.chunks:
            combined.append(chunk.source_file)
    combined_text = " ".join(combined).lower()
    if any(kw.lower() in combined_text for kw in expected_keywords):
        return True

    q = pipe["embedder"].embed(question).astype(np.float32)
    q /= np.linalg.norm(q) or 1.0
    centroids = pipe["registry"].get_all_centroids()
    for bid in bucket_ids:
        c = centroids.get(bid)
        if c is None:
            continue
        sim = float(q @ c.astype(np.float32))
        if sim > 0.5:
            return True
    return False


def _grounding_score(answer: str, answer_keywords: List[str]) -> bool:
    """At least 50% of expected answer keywords appear in the answer."""
    if not answer_keywords:
        return False
    ans = answer.lower()
    hits = sum(1 for kw in answer_keywords if kw.lower() in ans)
    return hits >= len(answer_keywords) / 2.0


def _cosine_score(embedder, answer: str, ground_truth: str) -> float:
    """sentence-transformer cosine sim between answer and ground-truth."""
    if not answer.strip():
        return 0.0
    a = embedder.embed(answer).astype(np.float32)
    b = embedder.embed(ground_truth).astype(np.float32)
    a /= np.linalg.norm(a) or 1.0
    b /= np.linalg.norm(b) or 1.0
    return float(a @ b)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_fixtures(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["fixtures"]


async def _run_eval_one_repo(name: str, repo_path: Path, fixtures_path: Path) -> dict:
    print(f"\n[eval] === {name} ({repo_path}) ===", file=sys.stderr, flush=True)
    print(f"[eval] loading pipeline…", file=sys.stderr, flush=True)
    pipe = _build_pipeline(repo_path)
    fixtures = _load_fixtures(fixtures_path)
    print(f"[eval] {len(fixtures)} fixtures loaded", file=sys.stderr, flush=True)

    # Accumulators per path. Phase 3-A diagnostic paths (latent_clean,
    # hybrid_clean) dropped — diagnosis done, LoRA confirmed essential.
    # text_lora dropped — diagnosis confirmed it's OOD on plain text.
    # 4 paths × 30 fixtures keeps eval ~1h 20min on MPS.
    paths = [
        "latent", "hybrid", "cache_aug",
        "cache_aug_no_verbatim",    # 4-C.6 ablation: latent-alone gate
        "cache_aug_greedy_nogate",  # 4-C.6 ablation: Cold Stop vs. plain greedy
        "text_clean", "no_context",
    ]
    grounding = {p: 0 for p in paths}
    cosine = {p: [] for p in paths}
    routing_hits = 0  # shared by latent + text paths
    multi_grounding = {p: 0 for p in paths}
    multi_total = 0
    per_q: list[dict] = []

    for i, fx in enumerate(fixtures, start=1):
        qid = fx["id"]
        question = fx["question"]
        is_multi = bool(fx.get("needs_multi_bucket", False))
        if is_multi:
            multi_total += 1

        print(f"[eval] [{i:2d}/{len(fixtures)}] {qid}: {question[:60]}…", file=sys.stderr, flush=True)

        # 1. Latent path (one orchestrator query; reps + query_embedding
        #    reused by hybrid / text paths).
        latent_answer, bucket_ids, reps, query_embedding = await _path_latent(pipe, question)
        route_ok = _routing_score(pipe, bucket_ids, fx["expected_bucket_keywords"], question)
        if route_ok:
            routing_hits += 1

        # 2. Hybrid path (latent + verbatim, LoRA on).
        hybrid_answer = await _path_hybrid(pipe, question, bucket_ids, reps, query_embedding)

        # 3. Cache-aug path (Phase 4-C). "" when state_dict is absent.
        cache_aug_answer = _path_cache_aug(pipe, question, bucket_ids, query_embedding)

        # 3b/3c. 4-C.6 fairness ablations (same 3B bundle, no extra model load).
        cache_aug_no_verb_answer = _path_cache_aug(
            pipe, question, bucket_ids, query_embedding, use_verbatim=False)
        cache_aug_greedy_answer = _path_cache_aug(
            pipe, question, bucket_ids, query_embedding, cold_stop_entropy=None)

        # 4. Text-clean (LoRA disabled). Same bucket_ids from latent route.
        text_clean_answer = _path_text(pipe, question, bucket_ids, lora_active=False)

        # 5. No-context (LoRA disabled).
        no_ctx_answer = _path_no_context(pipe, question)

        answers = {
            "latent": latent_answer,
            "hybrid": hybrid_answer,
            "cache_aug": cache_aug_answer,
            "cache_aug_no_verbatim": cache_aug_no_verb_answer,
            "cache_aug_greedy_nogate": cache_aug_greedy_answer,
            "text_clean": text_clean_answer,
            "no_context": no_ctx_answer,
        }

        row = {"id": qid, "question": question, "route_ok": route_ok, "multi": is_multi, "answers": {}}
        for p, ans in answers.items():
            ground = _grounding_score(ans, fx["answer_keywords"])
            cos = _cosine_score(pipe["embedder"], ans, fx["ground_truth_answer"])
            if ground:
                grounding[p] += 1
                if is_multi:
                    multi_grounding[p] += 1
            cosine[p].append(cos)
            row["answers"][p] = {"text": ans, "grounded": ground, "cos": cos}
        per_q.append(row)

    # Summary
    n = len(fixtures)
    print(f"\n[eval] === RESULTS ({name}) ===", file=sys.stderr, flush=True)
    print(f"[eval] routing: {routing_hits}/{n}", file=sys.stderr, flush=True)
    for p in paths:
        cos_mean = sum(cosine[p]) / max(1, len(cosine[p]))
        print(
            f"[eval] {p:12s}: grounding {grounding[p]:2d}/{n} "
            f"(multi {multi_grounding[p]}/{multi_total})  cos {cos_mean:.3f}",
            file=sys.stderr, flush=True,
        )

    return {
        "repo": name,
        "n_fixtures": n,
        "routing_hits": routing_hits,
        "grounding": grounding,
        "multi_grounding": multi_grounding,
        "multi_total": multi_total,
        "cosine_means": {p: (sum(cosine[p]) / max(1, len(cosine[p]))) for p in paths},
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# Pytest entry point
# ---------------------------------------------------------------------------

@pytest.mark.eval
async def test_eval_latent_vs_baseline():
    """Run the 4-path eval on every configured repo. Prints summary to stderr.

    Always passes — this is a measurement, not a pass/fail gate. The numbers
    in stderr are what we use to decide whether Phase 1's recipe fix worked.
    A separate gate test can be added later to enforce thresholds.
    """
    all_results = []
    for name, repo_path, fixtures_path in _REPOS:
        if not (repo_path / ".libucks").is_dir():
            pytest.skip(f"{name}: .libucks/ not initialised at {repo_path}")
        if not fixtures_path.exists():
            pytest.skip(f"{name}: fixtures missing at {fixtures_path}")
        result = await _run_eval_one_repo(name, repo_path, fixtures_path)
        all_results.append(result)

    # Persist the results so we can diff against future runs.
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latest.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[eval] results written to {out_path}", file=sys.stderr, flush=True)
