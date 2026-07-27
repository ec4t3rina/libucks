"""Best-epoch retention — CM-B.0b shipped a cartridge worse than one it trained.

bc6b90e2's epoch means were 5.4433, **4.1816**, 5.2721, 5.2284. `distill_bucket`
overwrites `checkpoint_path` unconditionally every epoch, so the run promoted
epoch 3 and epoch 1's weights were gone. This pins an opt-in `best_path` that
retains the best epoch alongside the rolling resume checkpoint.

The two files are deliberately separate: `checkpoint_path` must keep advancing
every epoch or the sidecar's `epochs_done` would no longer describe the saved
weights and resume would silently rewind.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer


def _tiny_trainer() -> CartridgeTrainer:
    model = torch.nn.Linear(2, 2)
    tok = SimpleNamespace(eos_token_id=0)
    return CartridgeTrainer(model, tok)


def _tiny_cartridge() -> KVPrefixCartridge:
    return KVPrefixCartridge(
        n_layers=2, n_kv_heads=2, prefix_len=4, head_dim=8, dtype=torch.float32,
    )


def _stub_with_kls(trainer: CartridgeTrainer, monkeypatch, per_epoch_kl: list[float],
                   queries_per_epoch: int) -> None:
    """Drive a deterministic mean_kl per epoch so best-epoch logic is testable."""
    monkeypatch.setattr(
        trainer, "_teacher_generate",
        lambda verbatim, q: (torch.tensor([1, 2], dtype=torch.long), None),
    )
    calls = {"n": 0}

    def _step(*a, **k):
        ep = calls["n"] // queries_per_epoch
        calls["n"] += 1
        kl = per_epoch_kl[min(ep, len(per_epoch_kl) - 1)]
        return {"kl": kl, "loss": kl, "lr": 0.0, "n_ans": 2}

    monkeypatch.setattr(trainer, "_step_cached", _step)


class TestBestPathIsOptIn:
    def test_absent_by_default_behaviour_is_unchanged(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [3.0, 1.0, 2.0], 1)
        ckpt = tmp_path / "b.ckpt.safetensors"
        res = trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                                     checkpoint_path=ckpt)
        assert not (tmp_path / "b.ckpt.safetensors.best").exists()
        assert "best_epoch" not in res or res["best_epoch"] is None

    def test_rolling_checkpoint_still_advances_every_epoch(self, tmp_path, monkeypatch):
        """Resume correctness must not regress when best_path is used."""
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [3.0, 1.0, 2.0], 1)
        ckpt = tmp_path / "b.ckpt.safetensors"
        best = tmp_path / "b.best.safetensors"
        trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                               checkpoint_path=ckpt, best_path=best)
        side = json.loads((tmp_path / "b.ckpt.safetensors.json").read_text())
        assert side["epochs_done"] == 3, "rolling checkpoint must describe all 3 epochs"


class TestBestEpochRetention:
    def test_best_file_written(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [3.0, 1.0, 2.0], 1)
        best = tmp_path / "b.best.safetensors"
        trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                               checkpoint_path=tmp_path / "c.safetensors",
                               best_path=best)
        assert best.exists()

    def test_reports_the_argmin_epoch(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [3.0, 1.0, 2.0], 1)
        res = trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                                     checkpoint_path=tmp_path / "c.safetensors",
                                     best_path=tmp_path / "b.best.safetensors")
        assert res["best_epoch"] == 1
        assert res["best_mean_kl"] == 1.0

    def test_the_cm_b_0b_shape_is_caught(self, tmp_path, monkeypatch):
        """bc6b90e2's actual curve: best is epoch 1, last is worse."""
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [5.4433, 4.1816, 5.2721, 5.2284], 1)
        res = trainer.distill_bucket(cart, "v", ["q"], epochs=4, lr=1e-2,
                                     checkpoint_path=tmp_path / "c.safetensors",
                                     best_path=tmp_path / "b.best.safetensors")
        assert res["best_epoch"] == 1
        assert res["best_mean_kl"] == 4.1816
        assert res["final_mean_kl"] > res["best_mean_kl"], (
            "this is the regression: the last epoch is worse than the best"
        )

    def test_monotonic_improvement_makes_best_the_last_epoch(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [4.0, 3.0, 2.0, 1.0], 1)
        res = trainer.distill_bucket(cart, "v", ["q"], epochs=4, lr=1e-2,
                                     checkpoint_path=tmp_path / "c.safetensors",
                                     best_path=tmp_path / "b.best.safetensors")
        assert res["best_epoch"] == 3

    def test_best_not_rewritten_when_epoch_is_worse(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [1.0, 9.0, 9.0], 1)
        best = tmp_path / "b.best.safetensors"
        saves: list[str] = []
        orig = cart.save
        monkeypatch.setattr(cart, "save",
                            lambda p: (saves.append(str(p)), orig(p)))
        trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                               checkpoint_path=tmp_path / "c.safetensors",
                               best_path=best)
        assert saves.count(str(best)) == 1, "best written once, at epoch 0 only"

    def test_best_sidecar_records_which_epoch_won(self, tmp_path, monkeypatch):
        trainer, cart = _tiny_trainer(), _tiny_cartridge()
        _stub_with_kls(trainer, monkeypatch, [3.0, 1.0, 2.0], 1)
        best = tmp_path / "b.best.safetensors"
        trainer.distill_bucket(cart, "v", ["q"], epochs=3, lr=1e-2,
                               checkpoint_path=tmp_path / "c.safetensors",
                               best_path=best)
        meta = json.loads((tmp_path / "b.best.safetensors.json").read_text())
        assert meta["best_epoch"] == 1
        assert meta["best_mean_kl"] == 1.0
