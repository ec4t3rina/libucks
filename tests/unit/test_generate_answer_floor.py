"""A no-context floor must share the cartridge path's decode loop exactly.

CM-B.0d leaves the cartridge scoring 0.67/8 and 4.33/16. Neither number means
anything without knowing what the same model says with NO memory at all — if the
floor is also ~4/16, the latent channel is inert and the extension fixtures are
answerable by guesswork.

The floor is therefore only valid if it differs from the cartridge run in exactly
one respect: the absence of the prefix. Any divergence in tokenisation, greedy
selection, EOS handling or attention-mask construction would make the comparison
worthless, so the floor reuses `generate_answer` with `cartridge=None` rather
than reimplementing the loop.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from libucks.cache_augmentation.cartridge import KVPrefixCartridge
from libucks.thinking.training.cartridge_trainer import CartridgeTrainer


class _Tok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        self.last_text = text
        return {"input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "ANS:" + ",".join(str(i) for i in ids)


class _Model(torch.nn.Module):
    """Emits a fixed token sequence, recording whether a prefix cache arrived."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.saw_past = []
        self.attn_widths = []
        self._seq = [11, 12, 0]  # third is EOS
        self._i = 0

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                use_cache=None, **kw):
        self.saw_past.append(past_key_values is not None)
        self.attn_widths.append(int(attention_mask.shape[1]))
        tok = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        logits = torch.full((1, input_ids.shape[1], 32), -1e9)
        logits[0, -1, tok] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _trainer(model):
    return CartridgeTrainer(model, _Tok())


def _cart():
    return KVPrefixCartridge(n_layers=1, n_kv_heads=1, prefix_len=4, head_dim=2,
                             dtype=torch.float32)


class TestSignature:
    def test_cartridge_parameter_accepts_none(self):
        sig = inspect.signature(CartridgeTrainer.generate_answer)
        assert "cartridge" in sig.parameters

    def test_floor_call_does_not_raise(self):
        m = _Model()
        assert isinstance(_trainer(m).generate_answer(None, "q?"), str)


class TestFloorSkipsThePrefixOnly:
    def test_no_prefix_cache_is_passed_when_cartridge_is_none(self):
        m = _Model()
        _trainer(m).generate_answer(None, "q?", max_new_tokens=4)
        assert m.saw_past[0] is False, "floor must not attach any prefix cache"

    def test_prefix_cache_is_passed_when_a_cartridge_is_given(self):
        m = _Model()
        _trainer(m).generate_answer(_cart(), "q?", max_new_tokens=4)
        assert m.saw_past[0] is True

    def test_attention_width_excludes_prefix_len_on_the_floor(self):
        """With P=4 and 3 query tokens the cartridge run starts at width 7; the
        floor must start at 3, or the mask silently attends to absent positions."""
        mc, mf = _Model(), _Model()
        _trainer(mc).generate_answer(_cart(), "q?", max_new_tokens=1)
        _trainer(mf).generate_answer(None, "q?", max_new_tokens=1)
        assert mc.attn_widths[0] == 7
        assert mf.attn_widths[0] == 3

    def test_same_query_text_is_built_either_way(self):
        """The prompt must be identical, or the floor answers a different question."""
        tc, tf = _trainer(_Model()), _trainer(_Model())
        tc.generate_answer(_cart(), "what is X?", max_new_tokens=1)
        tf.generate_answer(None, "what is X?", max_new_tokens=1)
        assert tc.tokenizer.last_text == tf.tokenizer.last_text


class TestDecodeLoopIsShared:
    def test_eos_stops_generation_identically(self):
        mc, mf = _Model(), _Model()
        a = _trainer(mc).generate_answer(_cart(), "q?", max_new_tokens=10)
        b = _trainer(mf).generate_answer(None, "q?", max_new_tokens=10)
        # Same stub token stream -> same text, and EOS truncates both at 2 tokens.
        assert a == b == "ANS:11,12"

    def test_max_new_tokens_is_respected(self):
        m = _Model()
        out = _trainer(m).generate_answer(None, "q?", max_new_tokens=1)
        assert out == "ANS:11"
