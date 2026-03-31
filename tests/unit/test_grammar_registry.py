"""Tests for GrammarRegistry — Steps 3.1 (TDD: write first, watch fail)."""
import pytest
import respx
import httpx
from pathlib import Path
from unittest.mock import MagicMock, patch

from libucks.parsing.grammar_registry import GrammarRegistry, UnsupportedLanguageError


FAKE_SO_BYTES = b"\x7fELF\x02\x01\x01\x00"


# ---------------------------------------------------------------------------
# Unsupported extension
# ---------------------------------------------------------------------------

def test_unsupported_extension_raises(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)
    with pytest.raises(UnsupportedLanguageError) as exc:
        registry.get_parser(".xyz")
    assert ".xyz" in str(exc.value)


def test_unsupported_extension_message_contains_extension(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)
    with pytest.raises(UnsupportedLanguageError) as exc:
        registry.get_parser(".toml")
    assert ".toml" in str(exc.value)


def test_markdown_is_unsupported(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)
    with pytest.raises(UnsupportedLanguageError):
        registry.get_parser(".md")


# ---------------------------------------------------------------------------
# Download behaviour
# ---------------------------------------------------------------------------

@respx.mock
def test_first_call_triggers_download(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)

    route = respx.get(url__regex=r".*tree-sitter-python.*").mock(
        return_value=httpx.Response(200, content=FAKE_SO_BYTES)
    )

    with patch("libucks.parsing.grammar_registry.Language"), \
         patch("libucks.parsing.grammar_registry.Parser") as mock_parser_cls:
        mock_parser_cls.return_value = MagicMock()
        registry.get_parser(".py")

    assert route.called


@respx.mock
def test_second_call_does_not_redownload(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)

    route = respx.get(url__regex=r".*tree-sitter-python.*").mock(
        return_value=httpx.Response(200, content=FAKE_SO_BYTES)
    )

    mock_parser = MagicMock()
    with patch("libucks.parsing.grammar_registry.Language"), \
         patch("libucks.parsing.grammar_registry.Parser") as mock_parser_cls:
        mock_parser_cls.return_value = mock_parser
        registry.get_parser(".py")
        registry.get_parser(".py")

    assert route.call_count == 1


def test_cached_so_file_skips_download(tmp_path):
    """If the .so already exists in cache, no HTTP request should be made."""
    registry = GrammarRegistry(cache_dir=tmp_path)

    so_path = tmp_path / "python.so"
    so_path.write_bytes(FAKE_SO_BYTES)

    mock_parser = MagicMock()
    with patch("libucks.parsing.grammar_registry.Language"), \
         patch("libucks.parsing.grammar_registry.Parser") as mock_parser_cls:
        mock_parser_cls.return_value = mock_parser
        # No respx mock — any real HTTP call would raise ConnectionError
        result = registry.get_parser(".py")

    assert result is mock_parser


@respx.mock
def test_parser_is_returned(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)

    respx.get(url__regex=r".*tree-sitter-python.*").mock(
        return_value=httpx.Response(200, content=FAKE_SO_BYTES)
    )

    sentinel = MagicMock(name="parser-sentinel")
    with patch("libucks.parsing.grammar_registry.Language"), \
         patch("libucks.parsing.grammar_registry.Parser") as mock_parser_cls:
        mock_parser_cls.return_value = sentinel
        result = registry.get_parser(".py")

    assert result is sentinel


# ---------------------------------------------------------------------------
# Multiple extensions share cache
# ---------------------------------------------------------------------------

@respx.mock
def test_js_extension_triggers_different_download(tmp_path):
    registry = GrammarRegistry(cache_dir=tmp_path)

    py_route = respx.get(url__regex=r".*tree-sitter-python.*").mock(
        return_value=httpx.Response(200, content=FAKE_SO_BYTES)
    )
    js_route = respx.get(url__regex=r".*tree-sitter-javascript.*").mock(
        return_value=httpx.Response(200, content=FAKE_SO_BYTES)
    )

    with patch("libucks.parsing.grammar_registry.Language"), \
         patch("libucks.parsing.grammar_registry.Parser") as mock_parser_cls:
        mock_parser_cls.return_value = MagicMock()
        registry.get_parser(".py")
        registry.get_parser(".js")

    assert py_route.call_count == 1
    assert js_route.call_count == 1
