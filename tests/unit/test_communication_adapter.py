"""Phase 9-A: CommunicationAdapter — neural switchboard for Librarian tensors.

Uses small hidden_dim=64 so tests run on CPU without GPU memory pressure.
Production uses hidden_dim=2048 matching Qwen2.5-3B-Instruct.
"""
from __future__ import annotations

import pytest
import torch

from libucks.thinking.communication_adapter import CommunicationAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    """Small adapter suitable for CPU unit tests."""
    return CommunicationAdapter(hidden_dim=64, output_len=32)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestCommunicationAdapterShape:
    def test_three_variable_length_tensors_produce_correct_shape(self, adapter):
        reps = [torch.randn(10, 64), torch.randn(15, 64), torch.randn(8, 64)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_single_tensor_produces_correct_shape(self, adapter):
        reps = [torch.randn(10, 64)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_two_tensors_same_length_produce_correct_shape(self, adapter):
        reps = [torch.randn(5, 64), torch.randn(5, 64)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_very_short_sequence_length_one(self, adapter):
        reps = [torch.randn(1, 64)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_very_long_sequence_produces_correct_shape(self, adapter):
        reps = [torch.randn(512, 64)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_output_len_is_configurable(self):
        adapter_k16 = CommunicationAdapter(hidden_dim=64, output_len=16)
        reps = [torch.randn(10, 64)]
        output = adapter_k16(reps)
        assert output.shape == (16, 64)

    def test_output_len_default_is_32(self):
        adapter_default = CommunicationAdapter(hidden_dim=64)
        reps = [torch.randn(10, 64)]
        output = adapter_default(reps)
        assert output.shape[0] == 32

    def test_hidden_dim_propagates_to_output(self):
        adapter_128 = CommunicationAdapter(hidden_dim=128, output_len=8)
        reps = [torch.randn(5, 128)]
        output = adapter_128(reps)
        assert output.shape == (8, 128)


# ---------------------------------------------------------------------------
# Empty input guard
# ---------------------------------------------------------------------------

class TestCommunicationAdapterEmptyInput:
    def test_empty_list_raises_value_error(self, adapter):
        with pytest.raises(ValueError, match="non-empty"):
            adapter([])


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

class TestCommunicationAdapterDevice:
    def test_cpu_input_produces_cpu_output(self, adapter):
        reps = [torch.randn(5, 64)]
        output = adapter(reps)
        assert output.device.type == "cpu"

    def test_output_is_2d_tensor(self, adapter):
        reps = [torch.randn(10, 64), torch.randn(7, 64)]
        output = adapter(reps)
        assert output.dim() == 2


# ---------------------------------------------------------------------------
# Module properties
# ---------------------------------------------------------------------------

class TestCommunicationAdapterModule:
    def test_is_nn_module(self, adapter):
        import torch.nn as nn
        assert isinstance(adapter, nn.Module)

    def test_has_trainable_parameters(self, adapter):
        params = list(adapter.parameters())
        assert len(params) > 0

    def test_pool_query_is_learnable_parameter(self, adapter):
        param_names = [name for name, _ in adapter.named_parameters()]
        assert any("pool_query" in name for name in param_names)

    def test_output_queries_are_learnable_parameters(self, adapter):
        param_names = [name for name, _ in adapter.named_parameters()]
        assert any("output_queries" in name for name in param_names)

    def test_forward_does_not_raise_for_n_equals_5(self, adapter):
        reps = [torch.randn(10, 64) for _ in range(5)]
        output = adapter(reps)
        assert output.shape == (32, 64)

    def test_production_hidden_dim_2048(self):
        """Verify the production-scale adapter can be instantiated."""
        prod_adapter = CommunicationAdapter(hidden_dim=2048, output_len=32)
        assert prod_adapter is not None
        params = list(prod_adapter.parameters())
        assert len(params) > 0
