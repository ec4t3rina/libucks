"""CM-A.2 unit tests for distill_bucket orchestration — no 3B model load.

The heavy paths (_teacher_generate, _step_cached) are stubbed out; these tests
pin the orchestration contract added after the Jul-3 MPS hang lost a full run:
with checkpoint_path set, the cartridge is saved after EVERY epoch, so a
mid-training wedge loses at most one epoch instead of everything.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer


def _tiny_trainer() -> CartridgeTrainer:
    model = torch.nn.Linear(2, 2)  # supplies parameters() for device/dtype probes
    tok = SimpleNamespace(eos_token_id=0)
    return CartridgeTrainer(model, tok)


def _tiny_cartridge() -> KVPrefixCartridge:
    return KVPrefixCartridge(
        n_layers=2, n_kv_heads=2, prefix_len=4, head_dim=8, dtype=torch.float32,
    )


def _stub_heavy_paths(trainer: CartridgeTrainer, monkeypatch) -> None:
    monkeypatch.setattr(
        trainer, "_teacher_generate",
        lambda verbatim, q: (torch.tensor([1, 2], dtype=torch.long), None),
    )
    monkeypatch.setattr(
        trainer, "_step_cached",
        lambda *a, **k: {"kl": 1.0, "loss": 1.0, "lr": 0.0, "n_ans": 2},
    )


def test_distill_bucket_checkpoints_every_epoch(tmp_path, monkeypatch):
    trainer = _tiny_trainer()
    cart = _tiny_cartridge()
    _stub_heavy_paths(trainer, monkeypatch)

    ckpt = tmp_path / "b.cartridge.ckpt.safetensors"
    saves: list[str] = []
    orig_save = cart.save
    monkeypatch.setattr(
        cart, "save", lambda path: (saves.append(str(path)), orig_save(path)),
    )

    res = trainer.distill_bucket(
        cart, "verbatim text", ["q1", "q2"], epochs=3, lr=1e-2,
        checkpoint_path=ckpt,
    )
    assert saves == [str(ckpt)] * 3  # one save per epoch, not just at the end
    assert ckpt.exists()
    assert res["epoch_mean_kl"] == [1.0, 1.0, 1.0]


def test_distill_bucket_no_checkpoint_by_default(tmp_path, monkeypatch):
    trainer = _tiny_trainer()
    cart = _tiny_cartridge()
    _stub_heavy_paths(trainer, monkeypatch)

    saves: list[str] = []
    monkeypatch.setattr(cart, "save", lambda path: saves.append(str(path)))

    trainer.distill_bucket(cart, "verbatim text", ["q1"], epochs=2, lr=1e-2)
    assert saves == []
