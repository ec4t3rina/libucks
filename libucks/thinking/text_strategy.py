"""TextStrategy — V1 ThinkingStrategy implementation using the Anthropic API.

Design:
  encode()  → returns text unchanged (no embedding; routing uses EmbeddingService)
  reason()  → constructs a prompt and calls the Anthropic messages API (async)
  decode()  → returns result unchanged (it is already a str in V1)

The Anthropic client is dependency-injected to keep tests hermetic.
Use TextStrategy.from_env() in production; pass client= in tests.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic

from libucks.thinking.base import Representation, ThinkingStrategy

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024


class TextStrategy(ThinkingStrategy):
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def from_env(cls, model: str = _DEFAULT_MODEL) -> "TextStrategy":
        """Construct using the ANTHROPIC_API_KEY environment variable."""
        return cls(client=AsyncAnthropic(), model=model)

    # ------------------------------------------------------------------
    # ThinkingStrategy interface
    # ------------------------------------------------------------------

    async def encode(self, text: str) -> Representation:
        """V1: passthrough — routing embeddings come from EmbeddingService."""
        return text

    async def reason(self, query: str, context: str) -> Representation:
        """Call the Anthropic API and return the response text as a Representation."""
        if context:
            prompt = f"Context:\n{context}\n\nQuery: {query}"
        else:
            prompt = query

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    async def decode(self, result: Representation) -> str:
        """V1: passthrough — result is already a str."""
        return result  # type: ignore[return-value]
