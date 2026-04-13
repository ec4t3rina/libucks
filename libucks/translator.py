"""Translator — the ONLY component permitted to call ThinkingStrategy.decode().

Receives N latent Representations from N Librarians. Passes them through the
CommunicationAdapter to produce a single soft-prompt, then calls
strategy.decode() exactly once.

strategy.decode() is the sole natural-language output returned to the MCP Bridge.
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
        if adapter is not None:
            try:
                _device = strategy._mgr.device
                self._adapter = adapter.to(_device)
            except Exception:
                self._adapter = adapter
        else:
            self._adapter = None

    async def synthesize(self, query: str, representations: List[Representation]) -> str:
        if not representations:
            _log("synthesize: no representations — returning fallback message")
            return "No relevant context found in the memory store."

        shapes = [tuple(r.shape) for r in representations]
        _log(f"synthesize: {len(representations)} latent representations, shapes={shapes}")
        return await self._synthesize_latent(representations, query=query)

    async def _synthesize_latent(
        self, representations: List[Representation], query: str = ""
    ) -> str:
        shapes = [tuple(r.shape) for r in representations]
        _log(f"_synthesize_latent: {len(representations)} reps, shapes={shapes}")
        # .contiguous() before the adapter: MultiheadAttention on MPS hangs
        # on non-contiguous tensors produced by prior squeeze/expand operations.
        contiguous_reps = [r.contiguous().to(torch.float32) for r in representations]

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
        # This is the ONLY authorised call to decode() in the entire system.
        _log("_synthesize_latent: calling decode()")
        result = await self._strategy.decode(synthesized, query=query)
        _log(f"_synthesize_latent: decode complete ({len(result)} chars)")
        return result
