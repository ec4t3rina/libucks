"""Translator — the ONLY component permitted to call ThinkingStrategy.decode().

Receives N Representations from N Librarians plus the original query string.
Synthesizes a single coherent English answer and strips all metadata before
returning plain text to the MCP Bridge.
"""
from __future__ import annotations

from typing import List

from libucks.thinking.base import Representation, ThinkingStrategy


class Translator:
    def __init__(self, strategy: ThinkingStrategy) -> None:
        self._strategy = strategy

    async def synthesize(self, query: str, representations: List[Representation]) -> str:
        if not representations:
            return "No relevant context found in the memory store."

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
