"""Translator — the ONLY component permitted to call ThinkingStrategy.decode().

Receives N latent Representations from N Librarians. Passes them through the
CommunicationAdapter to produce a single soft-prompt, then calls
strategy.decode() exactly once.

strategy.decode() is the sole natural-language output returned to the MCP Bridge.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np
import torch

from libucks.thinking.base import Representation, ThinkingStrategy


def _log(msg: str) -> None:
    print(f"[libucks:translator] {msg}", file=sys.stderr, flush=True)


class Translator:
    def __init__(
        self,
        strategy: ThinkingStrategy,
        adapter: Optional[object] = None,
        store: Optional[object] = None,
        hybrid: bool = False,
        verbatim_max_chars: int = 3000,
        chunk_retriever: Optional[object] = None,
        *,
        cache_aug: Optional[dict] = None,
    ) -> None:
        self._strategy = strategy
        if adapter is not None:
            try:
                _device = strategy._mgr.device
                self._adapter = adapter.to(_device)
            except Exception:
                self._adapter = adapter
        else:
            self._adapter = None
        self._store = store
        self._hybrid = hybrid
        self._verbatim_max_chars = verbatim_max_chars
        self._chunk_retriever = chunk_retriever
        # cache_aug bundle: {"coproc", "fusion", "kv_cache", "receiver",
        # "tokenizer", "device", "bucket_chunks"}. None when cache_aug mode
        # is inactive. Set externally by the eval harness or MCP bridge.
        self._cache_aug = cache_aug

    async def synthesize(
        self,
        query: str,
        representations: List[Representation],
        bucket_ids: Optional[List[str]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        if not representations:
            _log("synthesize: no representations — returning fallback message")
            return "No relevant context found in the memory store."

        shapes = [tuple(r.shape) for r in representations]
        _log(f"synthesize: {len(representations)} latent representations, shapes={shapes}")
        return await self._synthesize_latent(
            representations,
            query=query,
            bucket_ids=bucket_ids,
            query_embedding=query_embedding,
        )

    def synthesize_cache_aug(
        self,
        query: str,
        bucket_ids: List[str],
        query_embedding: Optional[np.ndarray] = None,
        *,
        use_verbatim: bool = True,
        cold_stop_entropy: Optional[float] = 4.0,
    ) -> str:
        """Phase 4-C cache-augmentation decode path.

        Bypasses the CommunicationAdapter; instead, per-bucket KV caches feed
        the Coprocessor → CrossBucketFusion → augmented_decode (which is the
        sole decode point in this mode, preserving the faithfulness constraint
        on a per-architecture basis).
        """
        if self._cache_aug is None:
            raise RuntimeError(
                "synthesize_cache_aug called but cache_aug bundle was not provided"
            )
        from libucks.cache_augmentation.decode import build_z_fused, augmented_decode

        bundle = self._cache_aug
        z_fused = build_z_fused(
            bundle["coproc"], bundle["fusion"], bundle["kv_cache"],
            bucket_ids, bundle["bucket_chunks"], bundle["device"],
        )
        if z_fused is None:
            _log("synthesize_cache_aug: no fresh bucket caches — fallback message")
            return "No relevant context found in the memory store."

        # Phase 4-C.6 fairness ablations:
        #   use_verbatim=False  → isolate the latent (z_fused) contribution
        #                         ("latent alone is viable" gate).
        #   cold_stop_entropy=None → manual greedy WITHOUT the entropy gate,
        #                         to attribute the win to Cold Stop vs. greedy.
        verbatim = (
            self._gather_verbatim(bucket_ids, query_embedding=query_embedding)
            if use_verbatim
            else ""
        )
        return augmented_decode(
            bundle["receiver"], bundle["tokenizer"], z_fused,
            query=query, verbatim=verbatim,
            cold_stop_entropy=cold_stop_entropy,
        )

    def _gather_verbatim(
        self,
        bucket_ids: Optional[List[str]],
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        if not (self._hybrid and self._store is not None and bucket_ids):
            return ""
        per_bucket = max(200, self._verbatim_max_chars // max(1, len(bucket_ids)))
        chunks: list[str] = []
        for bid in bucket_ids:
            text = self._gather_for_bucket(bid, per_bucket, query_embedding)
            if text:
                chunks.append(text)
        verbatim = "\n\n---\n\n".join(chunks)
        if verbatim:
            _log(f"verbatim: {len(chunks)} chunks, {len(verbatim)} chars")
        return verbatim

    def _gather_for_bucket(
        self,
        bucket_id: str,
        max_chars: int,
        query_embedding: Optional[np.ndarray],
    ) -> str:
        """Per-bucket verbatim selection.

        With a ChunkRetriever + query_embedding, rerank chunks by query-cos
        and take the top scoring until the per-bucket char budget runs out.
        Falls back to positional `_collect_source_text` (first-N-chars-of-
        first-chunks) when either is absent — preserves prior behaviour
        for tests and any caller that doesn't pre-embed the query.
        """
        if self._chunk_retriever is not None and query_embedding is not None:
            from libucks.chunk_retriever import _read_chunk_content
            scored = self._chunk_retriever.score_chunks(bucket_id, query_embedding)
            parts: list[str] = []
            total = 0
            for meta, _score in scored:
                content = _read_chunk_content(meta)
                if not content:
                    continue
                block = f"# {meta.source_file}\n{content}\n"
                if total + len(block) > max_chars:
                    if not parts:
                        parts.append(block[:max_chars])
                    break
                parts.append(block)
                total += len(block)
            return "\n---\n\n".join(parts)

        from libucks.thinking.training.data_generator import _collect_source_text
        try:
            fm, _prose = self._store.read(bucket_id)
        except Exception as exc:
            _log(f"verbatim: skipping {bucket_id}: {exc}")
            return ""
        return _collect_source_text(fm, max_chars=max_chars)

    async def _synthesize_latent(
        self,
        representations: List[Representation],
        query: str = "",
        bucket_ids: Optional[List[str]] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> str:
        shapes = [tuple(r.shape) for r in representations]
        _log(f"_synthesize_latent: {len(representations)} reps, shapes={shapes}")
        # .contiguous() before the adapter: MultiheadAttention on MPS hangs
        # on non-contiguous tensors produced by prior squeeze/expand operations.
        # Cast to the adapter's dtype (not hardcoded float32) — on MPS the adapter
        # runs in float16 and a float32 input causes an mps_add broadcast crash.
        _target_dtype = (
            next(self._adapter.parameters()).dtype
            if self._adapter is not None
            else torch.float32
        )
        contiguous_reps = [r.contiguous().to(_target_dtype) for r in representations]

        if self._adapter is not None:
            with torch.no_grad():
                synthesized: torch.Tensor = self._adapter(contiguous_reps)
            _log(f"_synthesize_latent: adapter complete, output={tuple(synthesized.shape)}")
        else:
            # No adapter (e.g. during init before adapter training). Accept exactly
            # one Representation and decode it directly without merging.
            if len(contiguous_reps) != 1:
                raise ValueError(
                    f"Translator has no adapter: can only decode 1 Representation, "
                    f"got {len(contiguous_reps)}"
                )
            synthesized = contiguous_reps[0]
            _log("_synthesize_latent: no adapter — decoding single representation directly")
        verbatim = self._gather_verbatim(bucket_ids, query_embedding=query_embedding)
        # This is the ONLY authorised call to decode() in the entire system.
        _log("_synthesize_latent: calling decode()")
        result = await self._strategy.decode(synthesized, query=query, verbatim=verbatim)
        _log(f"_synthesize_latent: decode complete ({len(result)} chars)")
        return result
