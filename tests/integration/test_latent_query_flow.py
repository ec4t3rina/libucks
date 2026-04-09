"""Phase 9 integration: full latent pipeline — Librarians → Adapter → Translator.

Uses mock Librarians that return real torch.Tensor objects, a real
CommunicationAdapter (small dims), and a mock LatentStrategy decode.
Verifies that the full query flow produces a non-empty string without
any component calling strategy.decode() except the Translator.
"""
from __future__ import annotations

import pytest
import torch
from unittest.mock import AsyncMock, MagicMock

from libucks.thinking.communication_adapter import CommunicationAdapter
from libucks.translator import Translator


HIDDEN_DIM = 64   # small dim for CPU integration test
OUTPUT_LEN = 32


@pytest.fixture
def real_adapter():
    return CommunicationAdapter(hidden_dim=HIDDEN_DIM, output_len=OUTPUT_LEN)


@pytest.fixture
def latent_strategy():
    strategy = MagicMock()
    strategy.decode = AsyncMock(return_value="Decoded latent answer about the codebase.")
    return strategy


class TestLatentQueryFlow:
    async def test_full_pipeline_returns_string(self, real_adapter, latent_strategy):
        """Three Librarians return tensors → Adapter → Translator.decode() → str."""
        translator = Translator(latent_strategy, adapter=real_adapter)

        # Simulate three Librarians returning variable-length latent tensors
        librarian_outputs = [
            torch.randn(10, HIDDEN_DIM),
            torch.randn(15, HIDDEN_DIM),
            torch.randn(8, HIDDEN_DIM),
        ]

        result = await translator.synthesize("explain auth flow", librarian_outputs)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_adapter_output_shape_fed_to_decode(self, real_adapter, latent_strategy):
        """Verify the tensor passed to decode() has shape (output_len, hidden_dim)."""
        translator = Translator(latent_strategy, adapter=real_adapter)

        librarian_outputs = [torch.randn(12, HIDDEN_DIM), torch.randn(7, HIDDEN_DIM)]
        await translator.synthesize("query", librarian_outputs)

        decode_tensor = latent_strategy.decode.call_args[0][0]
        assert decode_tensor.shape == (OUTPUT_LEN, HIDDEN_DIM)

    async def test_decode_called_exactly_once_regardless_of_librarian_count(
        self, real_adapter, latent_strategy
    ):
        translator = Translator(latent_strategy, adapter=real_adapter)

        for n_librarians in [1, 3, 5]:
            latent_strategy.decode.reset_mock()
            reps = [torch.randn(10, HIDDEN_DIM) for _ in range(n_librarians)]
            await translator.synthesize("query", reps)
            latent_strategy.decode.assert_called_once()

    async def test_result_is_the_decoded_string(self, real_adapter, latent_strategy):
        latent_strategy.decode = AsyncMock(return_value="auth uses JWT middleware")
        translator = Translator(latent_strategy, adapter=real_adapter)

        result = await translator.synthesize(
            "how does auth work?",
            [torch.randn(10, HIDDEN_DIM)],
        )
        assert result == "auth uses JWT middleware"
