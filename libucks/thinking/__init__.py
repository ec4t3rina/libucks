"""Thinking — reasoning strategy interface and factory."""
from __future__ import annotations

from libucks.thinking.base import Representation, ThinkingStrategy


def create_strategy(config) -> ThinkingStrategy:
    """Construct the ThinkingStrategy specified by config.model.strategy."""
    if config.model.strategy == "latent":
        from libucks.thinking.model_manager import ModelManager
        from libucks.thinking.latent_strategy import LatentStrategy

        mgr = ModelManager()
        mgr.load(
            model_id=config.model.local_model,
            quantization=config.model.quantization,
            bnb_4bit_compute_dtype=config.model.bnb_4bit_compute_dtype,
            device=config.model.device,
        )
        return LatentStrategy(mgr)

    raise ValueError(f"Unknown strategy: {config.model.strategy!r}")
