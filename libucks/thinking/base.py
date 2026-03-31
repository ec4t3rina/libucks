"""ThinkingStrategy — abstract base class for all reasoning backends.

V1/V2 Migration Contract
------------------------
In V1, Representation is always str.
In V2, Librarians will return raw hidden-state tensors (torch.Tensor) from a
local model encoder.  The Translator is the ONLY component permitted to call
decode() and return its result as final natural-language output.

Representation is defined as Union[str, Any] here so that V2 can narrow it
to Union[str, torch.Tensor] without changing any call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

# V1: always str.  V2 will be Union[str, torch.Tensor].
Representation = Union[str, Any]


class ThinkingStrategy(ABC):
    """Abstract interface that decouples reasoning backends from components.

    Rule: Librarians call encode() and reason() only.
          Only the Translator calls decode() and returns its result as output.
    """

    @abstractmethod
    async def encode(self, text: str) -> Representation:
        """Encode text into a Representation.

        V1: passthrough (returns text unchanged).
        V2: runs text through the local model encoder → hidden-state tensor.
        """

    @abstractmethod
    async def reason(self, query: str, context: str) -> Representation:
        """Produce a Representation that answers query given context.

        V1: constructs a prompt → async API call → returns response str.
        V2: encodes query + context → model forward pass in latent space → tensor.
        """

    @abstractmethod
    async def decode(self, result: Representation) -> str:
        """Convert a Representation back to a human-readable string.

        V1: passthrough (result is already str).
        V2: runs the decoder head on the tensor → English str.

        ONLY the Translator is permitted to call this method and return
        its result as the final output to the MCP Bridge.
        """
