"""LatentStrategy — V2 stub.

All methods raise NotImplementedError.  This class exists so that the
ThinkingStrategy interface is exercised in tests from day one, and so that
type-checkers enforce the ABC contract even before V2 is implemented.

See ARCHITECTURE.md §4 for the V2 upgrade plan.
"""

from __future__ import annotations

from libucks.thinking.base import Representation, ThinkingStrategy

_MSG = (
    "V2 latent space reasoning is not yet implemented. "
    "See the V2 upgrade plan in ARCHITECTURE.md §4."
)


class LatentStrategy(ThinkingStrategy):
    async def encode(self, text: str) -> Representation:
        raise NotImplementedError(_MSG)

    async def reason(self, query: str, context: str) -> Representation:
        raise NotImplementedError(_MSG)

    async def decode(self, result: Representation) -> str:
        raise NotImplementedError(_MSG)
