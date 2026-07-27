"""Reproducibility of the cartridge measurement track.

Sweep finding (2026-07-28): NOTHING in the CM distill/eval pipeline is seeded.
`cartridge_trainer.py` shuffles the query order with the global `random` every
epoch, `KVPrefixCartridge.__init__` uses unseeded `torch.randn`, and neither
`cm_distill_buckets.py` nor `cm_eval_cartridge.py` calls any seeding function.
Only `scripts/diagnose_adapter.py` seeds anything, anywhere.

Why this outranks the individual bugs found so far: every headline in this
track is one sample compared against another sample — 2/8 vs 4/8, 7/25 vs
10/25, 5/8 vs 1/8 — and the run-to-run variance has never been measured, so
none of those deltas has a known error bar. Seeding does not by itself measure
the variance; it makes measuring it possible (vary the seed, hold all else).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer


def _trainer() -> CartridgeTrainer:
    return CartridgeTrainer(torch.nn.Linear(2, 2), SimpleNamespace(eos_token_id=0))


def _cart() -> KVPrefixCartridge:
    return KVPrefixCartridge(n_layers=2, n_kv_heads=2, prefix_len=4, head_dim=8,
                             dtype=torch.float32)


def _record_order(trainer, monkeypatch) -> list[str]:
    """Capture the query order the trainer actually steps through."""
    seen: list[str] = []
    monkeypatch.setattr(
        trainer, "_teacher_generate",
        lambda verbatim, q: (torch.tensor([1, 2], dtype=torch.long), None),
    )

    def _step(cartridge, optimizer, scheduler, verbatim, q, ans):
        seen.append(q)
        return {"kl": 1.0, "loss": 1.0, "lr": 0.0, "n_ans": 2}

    monkeypatch.setattr(trainer, "_step_cached", _step)
    return seen


QUERIES = [f"q{i}" for i in range(12)]


class TestSeededDistillIsReproducible:
    def test_same_seed_gives_the_same_query_order(self, monkeypatch):
        orders = []
        for _ in range(2):
            t = _trainer()
            seen = _record_order(t, monkeypatch)
            t.distill_bucket(_cart(), "v", QUERIES, epochs=3, lr=1e-2, seed=1234)
            orders.append(list(seen))
        assert orders[0] == orders[1], "a seeded run must replay identically"

    def test_different_seeds_give_different_orders(self, monkeypatch):
        orders = []
        for s in (1, 2):
            t = _trainer()
            seen = _record_order(t, monkeypatch)
            t.distill_bucket(_cart(), "v", QUERIES, epochs=3, lr=1e-2, seed=s)
            orders.append(list(seen))
        assert orders[0] != orders[1], (
            "distinct seeds must explore distinct orders, or varying the seed "
            "cannot measure run-to-run variance"
        )

    def test_epochs_within_a_run_still_differ(self, monkeypatch):
        """Seeding must not collapse every epoch onto one fixed order."""
        t = _trainer()
        seen = _record_order(t, monkeypatch)
        t.distill_bucket(_cart(), "v", QUERIES, epochs=3, lr=1e-2, seed=7)
        n = len(QUERIES)
        ep0, ep1 = seen[:n], seen[n:2 * n]
        assert sorted(ep0) == sorted(ep1), "each epoch must cover every query once"
        assert ep0 != ep1, "shuffling must still happen between epochs"

    def test_unseeded_remains_the_default(self, monkeypatch):
        """Omitting seed must not change the existing signature's behaviour."""
        t = _trainer()
        _record_order(t, monkeypatch)
        res = t.distill_bucket(_cart(), "v", QUERIES, epochs=1, lr=1e-2)
        assert res["n_queries"] == len(QUERIES)


class TestCartridgeGeometryIsReadable:
    """`PREFIX_LEN = 128` is duplicated across cm_distill_buckets and
    cm_eval_cartridge under reciprocal "MUST match" comments. cartridge.py's own
    load() docstring says a mismatch "is a question of when, not if". The
    geometry is already written into the file; it should be read, not restated.
    """

    def test_read_geometry_returns_what_save_wrote(self, tmp_path):
        cart = KVPrefixCartridge(n_layers=3, n_kv_heads=2, prefix_len=16,
                                 head_dim=8, dtype=torch.float32)
        p = tmp_path / "c.safetensors"
        cart.save(p)
        geo = KVPrefixCartridge.read_geometry(p)
        assert geo == {"n_layers": 3, "n_kv_heads": 2, "prefix_len": 16, "head_dim": 8}

    def test_prefix_len_is_recoverable_without_guessing(self, tmp_path):
        cart = KVPrefixCartridge(n_layers=2, n_kv_heads=2, prefix_len=384,
                                 head_dim=8, dtype=torch.float32)
        p = tmp_path / "big.safetensors"
        cart.save(p)
        assert KVPrefixCartridge.read_geometry(p)["prefix_len"] == 384

    def test_round_trips_into_a_matching_cartridge(self, tmp_path):
        src = KVPrefixCartridge(n_layers=2, n_kv_heads=2, prefix_len=16,
                                head_dim=8, dtype=torch.float32)
        p = tmp_path / "c.safetensors"
        src.save(p)
        geo = KVPrefixCartridge.read_geometry(p)
        dst = KVPrefixCartridge(dtype=torch.float32, **geo)
        dst.load(p)  # raises on any geometry mismatch
        assert torch.allclose(dst.k[0], src.k[0])

    def test_missing_metadata_raises_rather_than_guessing(self, tmp_path):
        from safetensors.torch import save_file

        p = tmp_path / "nometa.safetensors"
        save_file({"k_0": torch.zeros(1, 2, 4, 8)}, str(p))
        try:
            KVPrefixCartridge.read_geometry(p)
        except (ValueError, KeyError):
            return
        raise AssertionError("a file without geometry metadata must not be guessed at")
