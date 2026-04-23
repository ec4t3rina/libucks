"""ThinkingStrategy ABC and LatentStrategy subclass contract.

TextStrategy (V1) has been removed. These tests verify the abstract base
and that LatentStrategy correctly satisfies the interface.
"""
from __future__ import annotations

import pytest

from libucks.thinking.base import Representation, ThinkingStrategy
from libucks.thinking.latent_strategy import LatentStrategy


class TestThinkingStrategyABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ThinkingStrategy()  # type: ignore[abstract]

    def test_latent_strategy_is_subclass(self):
        assert issubclass(LatentStrategy, ThinkingStrategy)

    def test_representation_type_alias_exists(self):
        assert Representation is not None
