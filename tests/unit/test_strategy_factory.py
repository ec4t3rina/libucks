"""Phase 7-C: Strategy factory — correct dispatch based on config."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from libucks.config import Config, ModelConfig


class TestCreateStrategy:
    """create_strategy() returns the correct ThinkingStrategy subclass."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_default_config_uses_latent_strategy(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking import create_strategy
        from libucks.thinking.latent_strategy import LatentStrategy

        cfg = Config()  # default strategy="latent"
        strategy = create_strategy(cfg)
        assert isinstance(strategy, LatentStrategy)

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

        # from_pretrained is called for both model and tokenizer; check the model call
        model_call_args = mock_model_cls.from_pretrained.call_args_list[0]
        assert model_call_args[0][0] == "custom/my-model"

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_latent_strategy_has_model_manager(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking import create_strategy
        from libucks.thinking.latent_strategy import LatentStrategy

        cfg = Config(model=ModelConfig(strategy="latent"))
        strategy = create_strategy(cfg)
        assert isinstance(strategy, LatentStrategy)
        assert strategy._mgr is not None

    def test_unknown_strategy_raises_value_error(self):
        from libucks.thinking import create_strategy
        # Bypass ModelConfig validation by patching the config object
        cfg = Config()
        cfg.model.strategy = "unknown"
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_strategy(cfg)


class TestModelConfigValidation:
    """ModelConfig rejects invalid field values."""

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="strategy"):
            ModelConfig(strategy="invalid")

    def test_invalid_quantization_raises_value_error(self):
        with pytest.raises(ValueError, match="quantization"):
            ModelConfig(quantization="fp8")

    def test_valid_strategies_accepted(self):
        cfg = ModelConfig(strategy="latent")
        assert cfg.strategy == "latent"

    def test_text_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy must be 'latent'"):
            ModelConfig(strategy="text")

    def test_valid_quantizations_accepted(self):
        for q in ("none", "4bit", "8bit"):
            cfg = ModelConfig(quantization=q)
            assert cfg.quantization == q

    def test_default_strategy_is_latent(self):
        cfg = ModelConfig()
        assert cfg.strategy == "latent"

    def test_base_model_field_exists(self):
        cfg = ModelConfig()
        assert cfg.base_model == "Qwen/Qwen2.5-0.5B"
