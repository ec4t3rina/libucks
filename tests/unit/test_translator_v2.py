"""Phase 9-B: Translator V2 — latent path dispatch and decode boundary.

Verifies that:
  - str representations → existing V1 text path (unchanged)
  - tensor representations → new V2 latent path
  - strategy.decode() is called exactly once in both paths
  - CommunicationAdapter is called only in the latent path
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
    return adapter


# ---------------------------------------------------------------------------
# V1 text path (backward compatibility)
# ---------------------------------------------------------------------------

class TestTranslatorV1TextPath:
    async def test_str_representations_use_text_path(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        await translator.synthesize("query", ["ans1", "ans2", "ans3"])
        mock_adapter.assert_not_called()

    async def test_str_path_calls_strategy_reason(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        await translator.synthesize("query", ["ans1"])
        mock_strategy.reason.assert_called_once()

    async def test_str_path_calls_strategy_decode_once(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        await translator.synthesize("query", ["ans1"])
        mock_strategy.decode.assert_called_once()

    async def test_str_path_returns_decoded_string(self, mock_strategy):
        mock_strategy.decode = AsyncMock(return_value="V1 answer")
        translator = Translator(mock_strategy)
        result = await translator.synthesize("query", ["context"])
        assert result == "V1 answer"

    async def test_empty_representations_returns_fallback(self, mock_strategy):
        translator = Translator(mock_strategy)
        result = await translator.synthesize("query", [])
        assert "No relevant context" in result
        mock_strategy.reason.assert_not_called()
        mock_strategy.decode.assert_not_called()

    async def test_no_adapter_injected_still_works_for_text(self, mock_strategy):
        translator = Translator(mock_strategy)  # no adapter
        result = await translator.synthesize("query", ["text answer"])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# V2 latent path
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

    async def test_latent_path_empty_list_returns_fallback(self, mock_strategy, mock_adapter):
        translator = Translator(mock_strategy, adapter=mock_adapter)
        result = await translator.synthesize("query", [])
        assert "No relevant context" in result
        mock_adapter.assert_not_called()


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
