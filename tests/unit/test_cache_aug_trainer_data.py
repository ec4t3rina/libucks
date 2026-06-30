"""Unit tests for cache_aug_trainer.load_qa_pairs — specifically the stub filter."""
from __future__ import annotations

import json

from libucks.thinking.training.cache_aug_trainer import load_qa_pairs


def test_load_qa_pairs_drops_generic_stubs(tmp_path):
    """Stub pairs (Q="Explain concisely what this code does...") are filtered."""
    payload = {
        "bucket_a": {
            "pairs": [
                ["Explain concisely what this code does and how it works.", "BLUEPRINT"],
                ["What does the relay_tokens function do?", "It drops tokens with probability 0.4."],
            ],
        },
        "bucket_b": {
            "pairs": [
                ["Explain concisely what this code does and how it works.", "README"],
            ],
        },
    }
    qa_path = tmp_path / "qa_cache.json"
    qa_path.write_text(json.dumps(payload))

    samples = load_qa_pairs(qa_path)

    assert len(samples) == 1, "expected only the non-stub pair to survive"
    assert samples[0].bucket_id == "bucket_a"
    assert samples[0].query.startswith("What does the relay_tokens")


def test_load_qa_pairs_handles_empty_cache(tmp_path):
    qa_path = tmp_path / "qa_cache.json"
    qa_path.write_text("{}")
    assert load_qa_pairs(qa_path) == []


def test_load_qa_pairs_skips_malformed_pairs(tmp_path):
    payload = {
        "bucket_a": {
            "pairs": [
                ["only one element"],            # malformed: len < 2
                ["", "empty question"],          # empty Q
                ["empty answer", ""],            # empty A
                ["valid question", "valid answer"],
            ],
        },
    }
    qa_path = tmp_path / "qa_cache.json"
    qa_path.write_text(json.dumps(payload))
    samples = load_qa_pairs(qa_path)
    assert len(samples) == 1
    assert samples[0].query == "valid question"
