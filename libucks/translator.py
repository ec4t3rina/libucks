"""Translator — the ONLY component permitted to call ThinkingStrategy.decode().

Receives N Representations from N Librarians plus the original query string.
Selects the appropriate synthesis path based on representation type:

  V1 (str):    joins partial answers, calls strategy.reason() to synthesize,
               then strategy.decode() to produce the final string.
  V2 (tensor): passes the list of hidden-state tensors through the
               CommunicationAdapter to produce a single soft-prompt, then
               calls strategy.decode() exactly once.

In both paths, strategy.decode() is called exactly once and its return value
is the sole natural-language output returned to the MCP Bridge.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import torch

from libucks.thinking.base import Representation, ThinkingStrategy


def _log(msg: str) -> None:
    print(f"[libucks:translator] {msg}", file=sys.stderr, flush=True)


class Translator:
    def __init__(
        self,
        strategy: ThinkingStrategy,
        adapter: Optional[object] = None,
    ) -> None:
        self._strategy = strategy
        self._adapter = adapter

    async def synthesize(self, query: str, representations: List[Representation]) -> str:
        if not representations:
            _log("synthesize: no representations — returning fallback message")
            return "No relevant context found in the memory store."

        _log(f"synthesize: {len(representations)} representations, "
             f"type={'latent' if isinstance(representations[0], torch.Tensor) else 'text'}")

        if isinstance(representations[0], torch.Tensor):
            return await self._synthesize_latent(representations)

        return await self._synthesize_text(query, representations)

    # ------------------------------------------------------------------
    # V1 text path (unchanged)
    # ------------------------------------------------------------------

    async def _synthesize_text(
        self, query: str, representations: List[Representation]
    ) -> str:
        parts = "\n\n---\n\n".join(str(r) for r in representations)
        synthesis_prompt = (
            "You are synthesizing partial answers from multiple domain-specific memory buckets "
            "into a single, coherent response. Do not mention bucket IDs, internal metadata, "
            "or implementation details of the memory system. Answer directly and concisely.\n\n"
            f"Partial answers:\n{parts}"
        )
        combined: Representation = await self._strategy.reason(synthesis_prompt, query)
        # This is the ONLY authorised call to decode() in the entire system.
        return await self._strategy.decode(combined)

    # ------------------------------------------------------------------
    # V2 latent path
    # ------------------------------------------------------------------

    async def _synthesize_latent(
        self, representations: List[Representation]
    ) -> str:
        shapes = [tuple(r.shape) for r in representations]
        _log(f"_synthesize_latent: adapter forward, representations={shapes}")
        # .contiguous() before the adapter: MultiheadAttention on MPS hangs
        # on non-contiguous tensors produced by prior squeeze/expand operations.
        contiguous_reps = [r.contiguous() for r in representations]
        with torch.no_grad():
            synthesized: torch.Tensor = self._adapter(contiguous_reps)
        _log(f"_synthesize_latent: adapter complete, output={tuple(synthesized.shape)}")
        # This is the ONLY authorised call to decode() in the entire system.
        _log("_synthesize_latent: calling decode()")
        result = await self._strategy.decode(synthesized)
        _log(f"_synthesize_latent: decode complete ({len(result)} chars)")
        return result
