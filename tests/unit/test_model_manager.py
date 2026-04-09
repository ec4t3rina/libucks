"""Phase 7-B: ModelManager singleton — lifecycle and device detection."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestModelManagerLoad:
    """ModelManager.load() delegates to transformers correctly."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_calls_from_pretrained_with_model_id(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="none", device="cpu")

        mock_model_cls.from_pretrained.assert_called_once()
        call_args = mock_model_cls.from_pretrained.call_args
        assert call_args[0][0] == "Qwen/Qwen2.5-3B-Instruct"

        mock_tok_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-3B-Instruct"
        )

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_4bit_passes_quantization_config(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="4bit", device="cpu")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_8bit_passes_quantization_config(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="8bit", device="cpu")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_none_quantization_no_quantization_config(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "quantization_config" not in call_kwargs

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_calls_model_eval(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")

        mock_model.eval.assert_called_once()


class TestModelManagerCache:
    """get_model() and get_tokenizer() return cached instances."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_get_model_returns_cached(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")

        m1 = mgr.get_model()
        m2 = mgr.get_model()
        assert m1 is m2
        assert mock_model_cls.from_pretrained.call_count == 1

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_get_tokenizer_returns_cached(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")

        t1 = mgr.get_tokenizer()
        t2 = mgr.get_tokenizer()
        assert t1 is t2
        assert mock_tok_cls.from_pretrained.call_count == 1

    def test_get_model_before_load_raises(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_model()

    def test_get_tokenizer_before_load_raises(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_tokenizer()


class TestModelManagerUnload:
    """unload() clears internal state."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_unload_clears_model(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")
        mgr.unload()

        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_model()

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_unload_clears_tokenizer(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")
        mgr.unload()

        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_tokenizer()

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_unload_clears_device(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")
        mgr.unload()

        with pytest.raises(RuntimeError, match="not loaded"):
            _ = mgr.device

    def test_unload_before_load_is_noop(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.unload()  # must not raise


class TestModelManagerDevice:
    """device property reflects configured device."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_device_returns_configured_value(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="cpu")
        assert mgr.device == "cpu"

    def test_device_before_load_raises(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = mgr.device

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_auto_device_resolves_to_string(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load("test-model", quantization="none", device="auto")
        assert mgr.device in ("cpu", "cuda", "mps")


class TestModelManagerBaseModel:
    """load_base_model() loads a separate model/tokenizer pair for the receiver role."""

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_base_model_calls_from_pretrained(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="none", device="cpu")

        mock_model_cls.from_pretrained.assert_called_once()
        assert mock_model_cls.from_pretrained.call_args[0][0] == "Qwen/Qwen2.5-3B"

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_load_base_model_calls_eval(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mgr = ModelManager()
        mgr.load_base_model("test-base", quantization="none", device="cpu")

        mock_model.eval.assert_called_once()

    def test_get_base_model_before_load_raises(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_base_model()

    def test_get_base_tokenizer_before_load_raises(self):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_base_tokenizer()

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_get_base_model_returns_cached(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load_base_model("test-base", quantization="none", device="cpu")

        m1 = mgr.get_base_model()
        m2 = mgr.get_base_model()
        assert m1 is m2
        assert mock_model_cls.from_pretrained.call_count == 1

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_encoder_and_decoder_are_independent(self, mock_model_cls, mock_tok_cls):
        """load() and load_base_model() can use different model IDs."""
        from libucks.thinking.model_manager import ModelManager
        from unittest.mock import MagicMock

        instruct_model = MagicMock(name="instruct")
        base_model = MagicMock(name="base")
        mock_model_cls.from_pretrained.side_effect = [instruct_model, base_model]

        mgr = ModelManager()
        mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="none", device="cpu")
        mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="none", device="cpu")

        assert mgr.get_model() is instruct_model
        assert mgr.get_base_model() is base_model
        assert mgr.get_model() is not mgr.get_base_model()

    @patch("libucks.thinking.model_manager.AutoTokenizer")
    @patch("libucks.thinking.model_manager.AutoModelForCausalLM")
    def test_unload_clears_base_model(self, mock_model_cls, mock_tok_cls):
        from libucks.thinking.model_manager import ModelManager

        mgr = ModelManager()
        mgr.load_base_model("test-base", quantization="none", device="cpu")
        mgr.unload()

        with pytest.raises(RuntimeError, match="not loaded"):
            mgr.get_base_model()
