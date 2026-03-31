"""GrammarRegistry — maps file extensions to tree-sitter parsers.

Lazily downloads compiled grammar .so files from the tree-sitter GitHub
releases API on first use and caches them under ``~/.libucks/grammars/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import httpx

try:
    from tree_sitter import Language, Parser
except ImportError:  # pragma: no cover
    Language = None  # type: ignore[assignment,misc]
    Parser = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Extension → language name mapping
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
}

# Download URLs for compiled grammars.
# Pattern: https://github.com/tree-sitter/tree-sitter-{lang}/releases/download/{ver}/{lang}.so
_GRAMMAR_VERSION = "v0.21.0"
_GRAMMAR_BASE = "https://github.com/tree-sitter/tree-sitter-{lang}/releases/download/{ver}/{lang}.so"

DOWNLOAD_URLS: Dict[str, str] = {
    lang: _GRAMMAR_BASE.format(lang=lang, ver=_GRAMMAR_VERSION)
    for lang in set(SUPPORTED_LANGUAGES.values())
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsupportedLanguageError(Exception):
    """Raised when ``get_parser`` is called with an unsupported file extension."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class GrammarRegistry:
    """Thread-safe (for async) registry that provides tree-sitter parsers.

    Parameters
    ----------
    cache_dir:
        Directory where downloaded ``.so`` grammar files are cached.
        Defaults to ``~/.libucks/grammars/``.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".libucks" / "grammars"
        self._parsers: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_parser(self, extension: str) -> object:
        """Return a ``tree_sitter.Parser`` configured for *extension*.

        Downloads and caches the grammar ``.so`` on first call.

        Raises
        ------
        UnsupportedLanguageError
            If *extension* is not in ``SUPPORTED_LANGUAGES``.
        """
        ext = extension if extension.startswith(".") else f".{extension}"

        if ext not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"No tree-sitter grammar available for extension '{ext}'. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
            )

        language = SUPPORTED_LANGUAGES[ext]

        if language in self._parsers:
            return self._parsers[language]

        so_path = self._cache_dir / f"{language}.so"
        if not so_path.exists():
            self._download_grammar(language, so_path)

        parser = self._load_parser(language, so_path)
        self._parsers[language] = parser
        return parser

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_grammar(self, language: str, dest: Path) -> None:
        """Download the compiled grammar .so for *language* to *dest*."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = DOWNLOAD_URLS.get(language, _GRAMMAR_BASE.format(lang=language, ver=_GRAMMAR_VERSION))
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)

    def _load_parser(self, language: str, so_path: Path) -> object:
        """Load a ``tree_sitter.Parser`` from a compiled ``.so`` file."""
        if Language is None or Parser is None:
            raise RuntimeError(
                "tree-sitter is not installed. Install it with: pip install tree-sitter"
            )
        lang = Language(str(so_path), language)
        parser = Parser()
        parser.set_language(lang)
        return parser
