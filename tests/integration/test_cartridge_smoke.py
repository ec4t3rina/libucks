"""CM-A.1 integration smoke: context-distillation mechanics on a small model.

Validates the load-bearing correctness properties before the expensive 3B run:
  1. A distillation step runs end-to-end; KL is finite.
  2. Grads flow into the cartridge (its params move); the base model stays
     frozen (no param has requires_grad or a .grad).
  3. KL does not increase over a few epochs on a fixed (verbatim, query) — i.e.
     the cartridge is actually learning to reproduce the teacher.
  4. generate_answer produces text from the cartridge prefix alone.

Uses Qwen2.5-0.5B-Instruct (small, fast) — the trainer is model-agnostic
(cartridge geometry is read from the model config), so the 3B path exercises
the same code. Marked `smoke`; skips cleanly if the model can't be loaded.
"""
from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.smoke

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_VERBATIM = (
    "PANIC_RADIUS = 3  # hops a panic signal can spread\n"
    "def relay(agent, msg):\n"
    "    # Compliant agents relay with 80% probability, verbatim.\n"
    "    if agent.kind == 'Compliant' and random() < 0.8:\n"
    "        return msg\n"
    "    return None\n"
)
_QUERIES = [
    "What is PANIC_RADIUS?",
    "What probability does a Compliant agent relay with?",
    "What does the relay function return?",
]


@pytest.fixture(scope="module")
def small_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, dtype=torch.float32).eval().to(device)
        return model, tok, device
    except Exception as exc:  # offline / not cached
        pytest.skip(f"could not load {_MODEL_ID}: {exc}")


def test_distill_step_and_grad_isolation(small_model):
    from libucks.cache_augmentation.cartridge import KVPrefixCartridge
    from libucks.thinking.training.cartridge_trainer import CartridgeTrainer

    model, tok, device = small_model
    trainer = CartridgeTrainer(model, tok, max_answer_tokens=8, temperature=2.0)
    cart = KVPrefixCartridge.for_model(model, prefix_len=16, dtype=torch.float32).to(device)

    # Snapshot ALL params (per-layer grad is uneven — layer 0's key prefix can
    # get ~0 grad while deeper layers move; check aggregate movement).
    before = [p.detach().cpu().clone() for p in cart.parameters()]
    result = trainer.distill_bucket(cart, _VERBATIM, _QUERIES, epochs=3, lr=1e-2)

    # (1) finite KL history
    assert result["n_queries"] == len(_QUERIES)
    assert all(kl == kl for kl in result["epoch_mean_kl"])  # no NaN
    assert result["init_mean_kl"] > 0

    # (2) cartridge moved (aggregate); base model frozen
    total_delta = sum(
        (a - p.detach().cpu()).abs().sum().item()
        for a, p in zip(before, cart.parameters())
    )
    assert total_delta > 1e-4, f"cartridge did not update — grad/step broken (delta={total_delta})"
    for p in model.parameters():
        assert not p.requires_grad
        assert p.grad is None, "base model received gradients — receiver not frozen"

    # (3) KL did not increase (learning, not diverging)
    assert result["final_mean_kl"] <= result["init_mean_kl"] + 1e-3, (
        f"KL increased: init={result['init_mean_kl']:.4f} final={result['final_mean_kl']:.4f}"
    )

    # (4) latent-alone decode produces text
    ans = trainer.generate_answer(cart, _QUERIES[0], max_new_tokens=16)
    assert isinstance(ans, str)
