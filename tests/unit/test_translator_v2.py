"""Phase 9-B: Translator V2 — latent path dispatch and decode boundary.

Verifies that:
  - Tensor representations → V2 latent path through the CommunicationAdapter
  - strategy.decode() is called exactly once
  - With no adapter, a single representation is decoded directly (init use-case)
"""
from __future__ import annotations

import pytest
import torch
from unittest.mock import AsyncMock, MagicMock

from libucks.translator import Translator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_strategy():
    strategy = MagicMock()
    strategy.reason = AsyncMock(return_value="synthesized text")
    strategy.decode = AsyncMock(return_value="final decoded answer")
    return strategy


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.return_value = torch.randn(32, 2048)
    # Translator.__init__ calls adapter.to(device); make it return the adapter itself
    # so self._adapter is the same object the test has a reference to.
    adapter.to.return_value = adapter
    return adapter


# ---------------------------------------------------------------------------
# V2 latent path (with adapter)
# ---------------------------------------------------------------------------

class TestTranslatorV2LatentPath:
    async def test_tensor_representations_use_latent_path(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        reps = [torch.randn(10, 2048), torch.randn(15, 2048)]
        await translator.synthesize("query", reps)
        mock_adapter.assert_called_once()

    async def test_latent_path_passes_full_list_to_adapter(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        reps = [torch.randn(10, 2048), torch.randn(12, 2048), torch.randn(8, 2048)]
        await translator.synthesize("query", reps)

        adapter_call_args = mock_adapter.call_args[0][0]
        assert len(adapter_call_args) == 3
        assert all(isinstance(r, torch.Tensor) for r in adapter_call_args)

    async def test_latent_path_calls_decode_with_adapter_output(
        self, mock_strategy, mock_adapter
    ):
        sentinel = torch.randn(32, 2048)
        mock_adapter.return_value = sentinel

        translator = Translator(mock_strategy, adapter=mock_adapter)
        await translator.synthesize("query", [torch.randn(10, 2048)])

        decode_call_arg = mock_strategy.decode.call_args[0][0]
        assert torch.equal(decode_call_arg, sentinel)

    async def test_latent_path_calls_decode_exactly_once(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        reps = [torch.randn(10, 2048), torch.randn(10, 2048)]
        await translator.synthesize("query", reps)
        mock_strategy.decode.assert_called_once()

    async def test_latent_path_does_not_call_strategy_reason(
        self, mock_strategy, mock_adapter
    ):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        await translator.synthesize("query", [torch.randn(10, 2048)])
        mock_strategy.reason.assert_not_called()

    async def test_latent_path_returns_str_from_decode(self, mock_strategy, mock_adapter):
        mock_strategy.decode = AsyncMock(return_value="latent decoded answer")
        translator = Translator(mock_strategy, adapter=mock_adapter)
        result = await translator.synthesize("query", [torch.randn(10, 2048)])
        assert result == "latent decoded answer"

    async def test_empty_representations_returns_fallback(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        result = await translator.synthesize("query", [])
        assert "No relevant context" in result
        mock_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# No-adapter path (init / single-tensor use-case)
# ---------------------------------------------------------------------------

class TestTranslatorNoAdapterPath:
    async def test_single_rep_decoded_directly_when_no_adapter(self, mock_strategy):
        mock_strategy.decode = AsyncMock(return_value="decoded prose")
        translator = Translator(mock_strategy)  # no adapter
        result = await translator.synthesize("", [torch.randn(10, 2048)])
        assert result == "decoded prose"
        mock_strategy.decode.assert_called_once()

    async def test_multiple_reps_raise_without_adapter(self, mock_strategy):
        translator = Translator(mock_strategy)
        with pytest.raises(ValueError, match="no adapter"):
            await translator.synthesize("", [torch.randn(5, 2048), torch.randn(5, 2048)])

    async def test_empty_list_returns_fallback_without_adapter(self, mock_strategy):
        translator = Translator(mock_strategy)
        result = await translator.synthesize("", [])
        assert "No relevant context" in result
        mock_strategy.decode.assert_not_called()


# ---------------------------------------------------------------------------
# Translator construction
# ---------------------------------------------------------------------------

class TestTranslatorConstruction:
    def test_adapter_is_optional_defaults_to_none(self, mock_strategy):
        translator = Translator(mock_strategy)
        assert translator._adapter is None

    def test_adapter_is_stored_when_injected(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        assert translator._adapter is mock_adapter
