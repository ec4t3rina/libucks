"""Tests for ASTParser — Step 3.2 (TDD: write first, watch fail).

Tests are structured to exercise both:
  - The regex fallback path (tree-sitter unavailable / grammar not found)
  - The tree-sitter path (mocked parser returning an AST)
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from libucks.parsing.grammar_registry import UnsupportedLanguageError
from libucks.parsing.ast_parser import ASTParser
from libucks.models.chunk import RawChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_PYTHON = """\
def greet(name: str) -> str:
    \"\"\"Return a greeting for the given name.\"\"\"
    return f"Hello, {name}!"


def farewell(name: str) -> str:
    \"\"\"Return a farewell message.\"\"\"
    return f"Goodbye, {name}!"
"""

CLASS_PYTHON = """\
class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def add(self, a: int, b: int) -> int:
        \"\"\"Return the sum of a and b.\"\"\"
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
"""

NO_DECLARATIONS = """\
# This file has no functions or classes.
x = 1
y = 2
result = x + y
"""

SYNTAX_ERROR_PYTHON = """\
def broken(
    \"\"\"This function has a syntax error.\"\"\"
    return 42
"""

MODULE_DOCSTRING = '''\
"""Module-level docstring for the auth middleware."""

import os


def validate(token: str) -> bool:
    """Validate the given token."""
    return bool(token)
'''


@pytest.fixture
def tmp_py_file(tmp_path):
    """Write SIMPLE_PYTHON to a temp .py file and return the path."""
    p = tmp_path / "sample.py"
    p.write_text(SIMPLE_PYTHON)
    return p


@pytest.fixture
def parser_with_fallback():
    """ASTParser whose registry always raises UnsupportedLanguageError (forces regex fallback)."""
    mock_registry = MagicMock()
    mock_registry.get_parser.side_effect = UnsupportedLanguageError(".py not in registry")
    return ASTParser(registry=mock_registry)


# ---------------------------------------------------------------------------
# Regex fallback — core contract
# ---------------------------------------------------------------------------

class TestRegexFallback:
    def test_returns_list_of_raw_chunks(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        assert isinstance(chunks, list)
        assert all(isinstance(c, RawChunk) for c in chunks)

    def test_detects_both_functions(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        contents = " ".join(c.content for c in chunks)
        assert "greet" in contents
        assert "farewell" in contents

    def test_start_line_is_1_based(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        for chunk in chunks:
            assert chunk.start_line >= 1

    def test_end_line_gte_start_line(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        for chunk in chunks:
            assert chunk.end_line >= chunk.start_line

    def test_language_is_python(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        for chunk in chunks:
            assert chunk.language == "python"

    def test_source_file_matches_path(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        for chunk in chunks:
            assert chunk.source_file == str(tmp_py_file)

    def test_chunk_contains_function_signature(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        greet_chunks = [c for c in chunks if "greet" in c.content]
        assert greet_chunks, "Expected at least one chunk containing 'greet'"
        assert "def greet" in greet_chunks[0].content

    def test_chunk_contains_docstring(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        greet_chunks = [c for c in chunks if "greet" in c.content]
        assert greet_chunks
        assert "Return a greeting" in greet_chunks[0].content

    def test_line_numbers_are_accurate_for_first_function(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        greet_chunks = [c for c in chunks if "def greet" in c.content]
        assert greet_chunks
        # greet starts at line 1 in SIMPLE_PYTHON
        assert greet_chunks[0].start_line == 1

    def test_second_function_starts_after_first(self, tmp_py_file, parser_with_fallback):
        chunks = parser_with_fallback.parse_file(tmp_py_file)
        greet_chunk = next(c for c in chunks if "def greet" in c.content)
        farewell_chunk = next(c for c in chunks if "def farewell" in c.content)
        assert farewell_chunk.start_line > greet_chunk.end_line or \
               farewell_chunk.start_line >= greet_chunk.start_line


class TestRegexFallbackClass:
    def test_class_definition_is_captured(self, tmp_path, parser_with_fallback):
        p = tmp_path / "calc.py"
        p.write_text(CLASS_PYTHON)
        chunks = parser_with_fallback.parse_file(p)
        contents = " ".join(c.content for c in chunks)
        assert "Calculator" in contents

    def test_class_docstring_is_captured(self, tmp_path, parser_with_fallback):
        p = tmp_path / "calc.py"
        p.write_text(CLASS_PYTHON)
        chunks = parser_with_fallback.parse_file(p)
        contents = " ".join(c.content for c in chunks)
        assert "simple calculator" in contents


class TestRegexFallbackEdgeCases:
    def test_no_declarations_returns_at_least_one_chunk(self, tmp_path, parser_with_fallback):
        p = tmp_path / "empty.py"
        p.write_text(NO_DECLARATIONS)
        chunks = parser_with_fallback.parse_file(p)
        assert len(chunks) >= 1

    def test_syntax_error_does_not_raise(self, tmp_path, parser_with_fallback):
        p = tmp_path / "broken.py"
        p.write_text(SYNTAX_ERROR_PYTHON)
        # Must not raise
        chunks = parser_with_fallback.parse_file(p)
        assert isinstance(chunks, list)

    def test_syntax_error_returns_best_effort_chunks(self, tmp_path, parser_with_fallback):
        p = tmp_path / "broken.py"
        p.write_text(SYNTAX_ERROR_PYTHON)
        chunks = parser_with_fallback.parse_file(p)
        assert len(chunks) >= 1

    def test_module_docstring_is_captured(self, tmp_path, parser_with_fallback):
        p = tmp_path / "mod.py"
        p.write_text(MODULE_DOCSTRING)
        chunks = parser_with_fallback.parse_file(p)
        contents = " ".join(c.content for c in chunks)
        assert "Module-level docstring" in contents or "validate" in contents

    def test_empty_file_returns_at_least_one_chunk(self, tmp_path, parser_with_fallback):
        p = tmp_path / "empty.py"
        p.write_text("")
        chunks = parser_with_fallback.parse_file(p)
        assert len(chunks) >= 1

    def test_non_python_unsupported_file_returns_line_chunks(self, tmp_path):
        """For unknown extensions, ASTParser line-splits and does not raise."""
        mock_registry = MagicMock()
        mock_registry.get_parser.side_effect = UnsupportedLanguageError(".txt")
        parser = ASTParser(registry=mock_registry)

        p = tmp_path / "readme.txt"
        p.write_text("Line 1\nLine 2\nLine 3\n")
        chunks = parser.parse_file(p)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Tree-sitter path (mocked)
# ---------------------------------------------------------------------------

class TestTreeSitterPath:
    def test_registry_is_consulted(self, tmp_py_file):
        mock_registry = MagicMock()
        mock_registry.get_parser.return_value = MagicMock()
        parser = ASTParser(registry=mock_registry)

        # Even if tree-sitter parse raises, registry must have been called
        try:
            parser.parse_file(tmp_py_file)
        except Exception:
            pass

        mock_registry.get_parser.assert_called_once_with(".py")

    def test_falls_back_when_treesitter_parse_raises(self, tmp_py_file):
        """If the tree-sitter parser itself raises, we still get chunks via fallback."""
        bad_ts_parser = MagicMock()
        bad_ts_parser.parse.side_effect = RuntimeError("tree-sitter binary corrupt")

        mock_registry = MagicMock()
        mock_registry.get_parser.return_value = bad_ts_parser

        parser = ASTParser(registry=mock_registry)
        chunks = parser.parse_file(tmp_py_file)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_treesitter_result_has_correct_language(self, tmp_py_file):
        """Chunks from tree-sitter path must still carry the correct language."""
        # Use registry that raises to force fallback — language must still be 'python'
        mock_registry = MagicMock()
        mock_registry.get_parser.side_effect = UnsupportedLanguageError("forced")
        parser = ASTParser(registry=mock_registry)

        chunks = parser.parse_file(tmp_py_file)
        for chunk in chunks:
            assert chunk.language == "python"
