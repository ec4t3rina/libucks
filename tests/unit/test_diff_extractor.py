"""Phase B.1 Testing Gate — test_diff_extractor.py

Tests DiffExtractor._parse_diff_output() directly.
All git I/O is bypassed — git.Repo is mocked at construction time so no
real repository is required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from libucks.diff.diff_extractor import DiffExtractor

_REL = "click/core.py"


# ---------------------------------------------------------------------------
# Fixture: extractor with git.Repo stubbed out
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    with patch("libucks.diff.diff_extractor.git.Repo"):
        return DiffExtractor(Path("/fake/repo"))


# ---------------------------------------------------------------------------
# Canonical diff strings
# ---------------------------------------------------------------------------

def _added_only_diff() -> str:
    return (
        "--- a/click/core.py\n"
        "+++ b/click/core.py\n"
        "@@ -10,2 +10,3 @@\n"
        " existing_line\n"
        "+added_line\n"
        " another_line\n"
    )


def _removed_only_diff() -> str:
    return (
        "--- a/click/core.py\n"
        "+++ b/click/core.py\n"
        "@@ -10,3 +10,2 @@\n"
        " existing_line\n"
        "-removed_line\n"
        " another_line\n"
    )


def _mixed_diff() -> str:
    return (
        "--- a/click/core.py\n"
        "+++ b/click/core.py\n"
        "@@ -10,3 +10,3 @@\n"
        " existing_line\n"
        "-removed_line\n"
        "+added_line\n"
        " another_line\n"
    )


def _rename_diff() -> str:
    # @@ -1,2 +1,2 @@: source has 2 lines (1 context + 1 removed),
    # target has 2 lines (1 context + 1 added).
    return (
        "diff --git a/old_module.py b/new_module.py\n"
        "similarity index 90%\n"
        "rename from old_module.py\n"
        "rename to new_module.py\n"
        "--- a/old_module.py\n"
        "+++ b/new_module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context_line\n"
        "-old_content\n"
        "+new_content\n"
    )


def _binary_diff() -> str:
    return "Binary files a/image.png and b/image.png differ\n"


def _multi_hunk_diff() -> str:
    return (
        "--- a/click/core.py\n"
        "+++ b/click/core.py\n"
        "@@ -10,2 +10,3 @@\n"
        " line1\n"
        "+added1\n"
        " line2\n"
        "@@ -20,2 +21,3 @@\n"
        " line3\n"
        "+added2\n"
        " line4\n"
    )


# ---------------------------------------------------------------------------
# Empty and binary
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_diff_returns_empty_list(self, extractor):
        assert extractor._parse_diff_output("", _REL) == []

    def test_whitespace_only_diff_returns_empty_list(self, extractor):
        assert extractor._parse_diff_output("   \n  \n", _REL) == []

    def test_binary_diff_returns_empty_list(self, extractor):
        assert extractor._parse_diff_output(_binary_diff(), _REL) == []


# ---------------------------------------------------------------------------
# Added lines
# ---------------------------------------------------------------------------

class TestAddedLines:
    def test_returns_one_event(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert len(events) == 1

    def test_returns_one_hunk(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert len(events[0].hunks) == 1

    def test_added_line_in_added_lines(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert "added_line" in events[0].hunks[0].added_lines

    def test_removed_lines_empty(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].hunks[0].removed_lines == []


# ---------------------------------------------------------------------------
# Removed lines
# ---------------------------------------------------------------------------

class TestRemovedLines:
    def test_returns_one_event(self, extractor):
        events = extractor._parse_diff_output(_removed_only_diff(), _REL)
        assert len(events) == 1

    def test_removed_line_in_removed_lines(self, extractor):
        events = extractor._parse_diff_output(_removed_only_diff(), _REL)
        assert "removed_line" in events[0].hunks[0].removed_lines

    def test_added_lines_empty(self, extractor):
        events = extractor._parse_diff_output(_removed_only_diff(), _REL)
        assert events[0].hunks[0].added_lines == []


# ---------------------------------------------------------------------------
# Mixed hunk
# ---------------------------------------------------------------------------

class TestMixedHunk:
    def test_both_sides_populated(self, extractor):
        events = extractor._parse_diff_output(_mixed_diff(), _REL)
        hunk = events[0].hunks[0]
        assert "added_line" in hunk.added_lines
        assert "removed_line" in hunk.removed_lines


# ---------------------------------------------------------------------------
# Multi-hunk
# ---------------------------------------------------------------------------

class TestMultiHunk:
    def test_one_event_with_two_hunks(self, extractor):
        events = extractor._parse_diff_output(_multi_hunk_diff(), _REL)
        assert len(events) == 1
        assert len(events[0].hunks) == 2

    def test_each_hunk_has_correct_added_line(self, extractor):
        events = extractor._parse_diff_output(_multi_hunk_diff(), _REL)
        hunks = events[0].hunks
        assert "added1" in hunks[0].added_lines
        assert "added2" in hunks[1].added_lines


# ---------------------------------------------------------------------------
# Rename detection
# ---------------------------------------------------------------------------

class TestRenameDetection:
    def test_is_rename_true(self, extractor):
        events = extractor._parse_diff_output(_rename_diff(), "old_module.py")
        assert len(events) == 1
        assert events[0].is_rename is True

    def test_old_path_contains_old_module(self, extractor):
        events = extractor._parse_diff_output(_rename_diff(), "old_module.py")
        assert events[0].old_path is not None
        assert "old_module" in events[0].old_path

    def test_new_path_contains_new_module(self, extractor):
        events = extractor._parse_diff_output(_rename_diff(), "old_module.py")
        assert events[0].new_path is not None
        assert "new_module" in events[0].new_path

    def test_non_rename_diff_has_none_paths(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].old_path is None
        assert events[0].new_path is None


# ---------------------------------------------------------------------------
# Hunk line numbers and metadata
# ---------------------------------------------------------------------------

class TestHunkMetadata:
    def test_old_start_matches_diff_header(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].hunks[0].old_start == 10

    def test_new_start_matches_diff_header(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].hunks[0].new_start == 10

    def test_file_field_on_event_is_rel_path(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].file == _REL

    def test_file_field_on_hunk_is_rel_path(self, extractor):
        events = extractor._parse_diff_output(_added_only_diff(), _REL)
        assert events[0].hunks[0].file == _REL
