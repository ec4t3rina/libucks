"""ModelManager — singleton lifecycle for the local HuggingFace model.

Handles loading, caching, device selection, quantization, and teardown
of the transformer model used by LatentStrategy in V2.
"""
from __future__ import annotations

import gc

from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    """Manages a single local transformer model instance."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        # Separate pair for the Base receiver model (Interlat-Lite decoder)
        self._base_model = None
        self._base_tokenizer = None

    def load(
        self,
        model_id: str,
        quantization: str = "none",
        bnb_4bit_compute_dtype: str = "float32",
        device: str = "auto",
    ) -> None:
        """Load model and tokenizer from HuggingFace hub or local cache."""
        resolved_device = self._resolve_device(device)

        import torch

        _DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        _compute_dtype = _DTYPE_MAP.get(bnb_4bit_compute_dtype, torch.float32)

        kwargs: dict = {"device_map": resolved_device}

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_compute_dtype,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        _do_sequential_mps_quant = False
        if resolved_device == "mps":
            # Cap max_position_embeddings to 4096 BEFORE from_pretrained().
            # Qwen2.5's default is 32768; the causal attention mask during
            # model.generate() is (max_pos × max_pos × float32) =
            # 32768 × 32768 × 4 = exactly 2^32 bytes — Metal's hard per-NDArray
            # limit. 4096 keeps the mask at 4096 × 4096 × 2 ≈ 32 MB (float16).
            from transformers import AutoConfig
            mps_config = AutoConfig.from_pretrained(model_id)
            mps_config.max_position_embeddings = 4096
            kwargs["config"] = mps_config
            kwargs["attn_implementation"] = "eager"

            if quantization == "4bit":
                # quantize_model() spikes to 18 GB by holding FP16 + NF4 for
                # the whole model at once. Instead: load on CPU in FP16, then
                # convert one Linear at a time to Linear4bit and move each layer
                # to MPS before touching the next. Peak MPS stays at ~1 layer.
                del kwargs["quantization_config"]
                kwargs["device_map"] = "cpu"
                kwargs["torch_dtype"] = torch.float16
                _do_sequential_mps_quant = True
            else:
                # float16 for none/8bit paths
                if quantization == "none":
                    kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if _do_sequential_mps_quant:
            self._model = self._sequential_mps_quant(self._model, device=resolved_device)
        self._model.eval()

        if resolved_device == "mps":
            # Qwen2.5 ships generation_config.json with cache_implementation="static"
            # (added in transformers 4.45 for torch.compile performance).
            # StaticCache pre-allocates KV for the full max_position_embeddings=32768:
            #   36 layers × 2 × 8 kv_heads × 32768 × 128 × float16 ≈ 4.5 GB
            # This single allocation exceeds Metal's 2**32 byte per-NDArray limit and
            # crashes before the first token is decoded.
            # Setting cache_implementation=None forces DynamicCache, which grows
            # incrementally and never allocates more than the actual generation length.
            gen_cfg = getattr(self._model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "cache_implementation", None) == "static":
                gen_cfg.cache_implementation = None

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._device = resolved_device

    def load_base_model(
        self,
        model_id: str,
        quantization: str = "none",
        bnb_4bit_compute_dtype: str = "float32",
        device: str = "auto",
    ) -> None:
        """Load the Base receiver model for latent injection (Interlat-Lite decoder).

        Uses the same hardware-aware setup as load(), but stored separately so
        the Instruct encoder and Base receiver can differ in model_id.
        """
        resolved_device = self._resolve_device(device)

        import torch

        _DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        _compute_dtype = _DTYPE_MAP.get(bnb_4bit_compute_dtype, torch.float32)

        kwargs: dict = {"device_map": resolved_device}

        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_compute_dtype,
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        _do_sequential_mps_quant = False
        if resolved_device == "mps":
            from transformers import AutoConfig
            mps_config = AutoConfig.from_pretrained(model_id)
            mps_config.max_position_embeddings = 4096
            kwargs["config"] = mps_config
            kwargs["attn_implementation"] = "eager"

            if quantization == "4bit":
                del kwargs["quantization_config"]
                kwargs["device_map"] = "cpu"
                kwargs["torch_dtype"] = torch.float16
                _do_sequential_mps_quant = True
            else:
                if quantization == "none":
                    kwargs["torch_dtype"] = torch.float16

        self._base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if _do_sequential_mps_quant:
            self._base_model = self._sequential_mps_quant(self._base_model, device=resolved_device)
        self._base_model.eval()

        if resolved_device == "mps":
            gen_cfg = getattr(self._base_model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "cache_implementation", None) == "static":
                gen_cfg.cache_implementation = None

        self._base_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_base_model(self):
        """Return the cached Base receiver model. Raises RuntimeError if not loaded."""
        if self._base_model is None:
            raise RuntimeError("Base model not loaded — call load_base_model() first")
        return self._base_model

    def get_base_tokenizer(self):
        """Return the cached Base tokenizer. Raises RuntimeError if not loaded."""
        if self._base_tokenizer is None:
            raise RuntimeError("Base tokenizer not loaded — call load_base_model() first")
        return self._base_tokenizer

    def get_model(self):
        """Return the cached model. Raises RuntimeError if not loaded."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        return self._model

    def get_tokenizer(self):
        """Return the cached tokenizer. Raises RuntimeError if not loaded."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — call load() first")
        return self._tokenizer

    @property
    def hidden_dim(self) -> int:
        """Return the hidden_size of the loaded Instruct encoder model."""
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        return self._model.config.hidden_size

    @property
    def device(self) -> str:
        """Return the resolved device string. Raises RuntimeError if not loaded."""
        if self._device is None:
            raise RuntimeError("Device not loaded — call load() first")
        return self._device

    def unload_encoder(self) -> None:
        """Release only the Instruct encoder, keeping Base receiver in memory."""
        self._model = None
        self._tokenizer = None

    def unload(self) -> None:
        """Release all models and tokenizers, freeing memory."""
        self._model = None
        self._tokenizer = None
        self._device = None
        self._base_model = None
        self._base_tokenizer = None

    @staticmethod
    def _sequential_mps_quant(model: "AutoModelForCausalLM", device: str = "mps") -> "AutoModelForCausalLM":
        """Convert Linear layers to Linear4bit one at a time, flushing MPS between each.

        quantize_model() OOMs at 18 GB because it holds FP16 + NF4 for the
        entire model simultaneously. This method:
          1. Iterates every nn.Linear in model (loaded on CPU in FP16).
          2. Replaces it with a Linear4bit shell and moves that shell to MPS.
          3. Deletes the CPU FP16 module and flushes the MPS allocator pool.
        Peak MPS resident: ~one layer's worth of FP16 + NF4 at a time (~300 MB
        for Qwen2.5-3B's largest FFN projections) instead of 18 GB.

        Falls back gracefully: if mps_bitsandbytes is not installed or its
        Linear4bit API does not support direct weight assignment, the model is
        moved to MPS as-is in FP16 (same as the pre-quantization baseline).
        """
        import torch
        import torch.nn as nn

        try:
            from mps_bitsandbytes import Linear4bit
        except ImportError:
            # Library not installed — move whole model to MPS in FP16 (no quant)
            return model.to(device)

        def _get_parent_and_attr(root: nn.Module, full_name: str):
            """Return (parent_module, attribute_name) for a dotted module name."""
            parts = full_name.split(".")
            parent = root
            for part in parts[:-1]:
                parent = getattr(parent, part)
            return parent, parts[-1]

        # Collect only the names of Linear layers — do NOT hold module refs in
        # the list, or all FP16 weights stay pinned until the loop ends and we
        # gain nothing. Fetch each module fresh via getattr inside the loop so
        # the local var is the only reference and del actually frees it.
        # Exclude embed_tokens and lm_head: they must stay as standard
        # nn.Embedding / nn.Linear in float16 to avoid shape mismatches.
        _EXCLUDE = ("embed_tokens", "lm_head")
        linear_names = [
            name for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
            and not any(ext in name for ext in _EXCLUDE)
        ]

        for full_name in linear_names:
            parent, attr = _get_parent_and_attr(model, full_name)
            module = getattr(parent, attr)

            if not isinstance(module, nn.Linear):
                continue  # already replaced, skip

            try:
                # Prefer from_linear() — it handles weight copy + quant_state
                # population in one call (the canonical mps-bitsandbytes path).
                if hasattr(Linear4bit, "from_linear"):
                    q = Linear4bit.from_linear(
                        module,
                        quant_type="nf4",
                        compute_dtype=torch.float16,
                    )
                else:
                    # Fallback: manual construction then explicit pack
                    q = Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        quant_type="nf4",
                        compute_dtype=torch.float16,
                    )
                    with torch.no_grad():
                        q.weight = nn.Parameter(module.weight.detach().clone())
                        if module.bias is not None:
                            q.bias = nn.Parameter(module.bias.detach().clone())

                # Pack weights on CPU BEFORE moving to MPS so quant_state
                # metadata is fully populated and survives the device transfer.
                if hasattr(q, "pack_weights"):
                    q.pack_weights()
                elif hasattr(q, "quantize_"):
                    q.quantize_()

                # Now move the packed (NF4) tensor + quant_state to MPS
                q.to(device)

            except Exception:
                # API mismatch — plain FP16 fallback for this layer
                q = module.to(device)

            setattr(parent, attr, q)
            del module
            gc.collect()
            torch.mps.empty_cache()

        # Move all remaining non-quantized params (norms, embeddings, lm_head)
        # to MPS in one pass — they are small (~200 MB total for 3B models).
        # embed_tokens and lm_head are intentionally kept as float16 and moved
        # here; do NOT call tie_weights() after in-place layer replacement as
        # it can re-link lm_head to a now-quantized embed_tokens pointer.
        for name, param in list(model.named_parameters()):
            if param.device.type == "cpu":
                param.data = param.data.to(device)
        for name, buf in list(model.named_buffers()):
            if buf.device.type == "cpu":
                buf.data = buf.data.to(device)

        gc.collect()
        torch.mps.empty_cache()

        return model

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to the best available device (MPS → CUDA → CPU)."""
        if device != "auto":
            return device

        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
