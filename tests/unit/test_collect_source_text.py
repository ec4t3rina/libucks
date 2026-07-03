"""Tests for _collect_source_text slice-on-overflow behavior.

Bug: when a chunk's block exceeded the remaining max_chars budget the loop
broke without emitting anything, so a bucket whose FIRST chunk was larger
than max_chars returned "" (observed on echoswarm bucket 98ca5ddd at
max_chars=4096 during CM-A.2). Fix: slice the overflowing block to the
remaining budget instead of dropping it.

The function is duplicated in librarian.py and data_generator.py; both are
tested.
"""
from __future__ import annotations

import dataclasses

import pytest

from libucks import librarian
from libucks.thinking.training import data_generator


@dataclasses.dataclass
class _FakeChunk:
    source_file: str
    start_line: int
    end_line: int


@dataclasses.dataclass
class _FakeFrontMatter:
    chunks: list


def _make_chunk(tmp_path, name: str, n_lines: int, line: str = "x = 1") -> _FakeChunk:
    f = tmp_path / name
    f.write_text("\n".join(line for _ in range(n_lines)))
    return _FakeChunk(source_file=str(f), start_line=1, end_line=n_lines)


IMPLS = [librarian._collect_source_text, data_generator._collect_source_text]


@pytest.mark.parametrize("collect", IMPLS)
def test_oversized_first_chunk_returns_truncated_text(collect, tmp_path):
    """First chunk alone exceeds max_chars -> sliced, NOT empty."""
    fm = _FakeFrontMatter(chunks=[_make_chunk(tmp_path, "big.py", 500)])
    out = collect(fm, max_chars=200)
    assert out != ""
    assert len(out) <= 200
    assert "big.py" in out


@pytest.mark.parametrize("collect", IMPLS)
def test_fitting_chunks_are_kept_whole(collect, tmp_path):
    """Blocks within budget are concatenated unchanged (existing behavior)."""
    fm = _FakeFrontMatter(
        chunks=[
            _make_chunk(tmp_path, "a.py", 3),
            _make_chunk(tmp_path, "b.py", 3),
        ]
    )
    out = collect(fm, max_chars=10_000)
    assert "a.py" in out and "b.py" in out
    assert "\n---\n\n" in out


@pytest.mark.parametrize("collect", IMPLS)
def test_mid_list_overflow_slices_to_remaining_budget(collect, tmp_path):
    """Earlier full blocks are kept; the overflowing block fills the rest."""
    small = _make_chunk(tmp_path, "small.py", 3)
    big = _make_chunk(tmp_path, "big.py", 500)
    small_block_len = len(f"# {small.source_file}\n" + "x = 1\n" * 2 + "x = 1\n")
    # leave enough remaining budget for big's header line plus a few lines
    budget = small_block_len + len(f"# {big.source_file}\n") + 30
    fm = _FakeFrontMatter(chunks=[small, big])
    out = collect(fm, max_chars=budget)
    assert "small.py" in out
    assert "big.py" in out  # sliced fragment, previously dropped entirely
