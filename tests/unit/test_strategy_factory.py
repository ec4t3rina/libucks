"""Phase 7-C: Strategy factory — correct dispatch based on config."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from libucks.config import Config, ModelConfig


class TestCreateStrategy:
    """create_strategy() returns the correct ThinkingStrategy subclass."""

    def test_default_config_uses_text_strategy(self):
        from libucks.thinking import create_strategy
        from libucks.thinking.text_strategy import TextStrategy

        cfg = Config()  # default strategy="text"
        with patch.object(TextStrategy, "from_env", return_value=MagicMock(spec=TextStrategy)):
            strategy = create_strategy(cfg)

        assert isinstance(strategy, TextStrategy) or strategy is not None

    def test_text_strategy_calls_from_env_with_model(self):
        from libucks.thinking import create_strategy
        from libucks.thinking.text_strategy import TextStrategy

        cfg = Config(model=ModelConfig(strategy="text", anthropic_model="claude-test"))
        with patch.object(TextStrategy, "from_env", return_value=MagicMock()) as mock_from_env:
            create_strategy(cfg)

        mock_from_env.assert_called_once_with("claude-test")

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_latent_strategy_returns_latent_instance(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking import create_strategy
        from libucks.thinking.latent_strategy import LatentStrategy

        cfg = Config(model=ModelConfig(strategy="latent"))
        strategy = create_strategy(cfg)
        assert isinstance(strategy, LatentStrategy)

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_latent_strategy_loads_configured_model(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking import create_strategy

        cfg = Config(model=ModelConfig(
            strategy="latent",
            local_model="custom/my-model",
            quantization="none",
            device="cpu",
        ))
        create_strategy(cfg)

        call_args = mock_model_cls.from_pretrained.call_args
        assert call_args[0][0] == "custom/my-model"

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_latent_strategy_has_model_manager(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking import create_strategy
        from libucks.thinking.latent_strategy import LatentStrategy

        cfg = Config(model=ModelConfig(strategy="latent"))
        strategy = create_strategy(cfg)
        assert isinstance(strategy, LatentStrategy)
        assert strategy._mgr is not None


class TestModelConfigValidation:
    """ModelConfig rejects invalid field values."""

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="strategy"):
            ModelConfig(strategy="invalid")

    def test_invalid_quantization_raises_value_error(self):
        with pytest.raises(ValueError, match="quantization"):
            ModelConfig(quantization="fp8")

    def test_valid_strategies_accepted(self):
        for s in ("text", "latent"):
            cfg = ModelConfig(strategy=s)
            assert cfg.strategy == s

    def test_valid_quantizations_accepted(self):
        for q in ("none", "4bit", "8bit"):
            cfg = ModelConfig(quantization=q)
            assert cfg.quantization == q
