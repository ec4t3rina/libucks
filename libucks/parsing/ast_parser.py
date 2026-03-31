"""ASTParser — extracts RawChunk objects from source files.

Uses tree-sitter for accurate AST-based extraction and falls back to a
robust regex-based chunker when tree-sitter grammars are unavailable or
fail to load on the current OS.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from libucks.models.chunk import RawChunk
from libucks.parsing.grammar_registry import (
    GrammarRegistry,
    SUPPORTED_LANGUAGES,
    UnsupportedLanguageError,
)


# ---------------------------------------------------------------------------
# Regex patterns for the fallback chunker
# ---------------------------------------------------------------------------

# Matches a top-level or class-level function/class definition line.
_DEF_RE = re.compile(r"^(\s*)(def |class )", re.MULTILINE)

# Chunk size (lines) used when there are no recognisable declarations.
_FALLBACK_CHUNK_LINES = 50


class ASTParser:
    """Parses source files into ``RawChunk`` objects.

    Parameters
    ----------
    registry:
        ``GrammarRegistry`` to obtain tree-sitter parsers from.
        Defaults to a new ``GrammarRegistry()`` with the default cache dir.
    """

    def __init__(self, registry: Optional[GrammarRegistry] = None) -> None:
        self._registry = registry or GrammarRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, path: Path) -> List[RawChunk]:
        """Parse *path* and return a list of ``RawChunk`` objects.

        Falls back to a regex-based chunker when tree-sitter is unavailable
        or the grammar fails to load / parse.
        """
        extension = path.suffix.lower()
        language = SUPPORTED_LANGUAGES.get(extension, extension.lstrip(".") or "text")

        try:
            ts_parser = self._registry.get_parser(extension)
            return self._parse_with_treesitter(path, ts_parser, language)
        except (UnsupportedLanguageError, Exception):
            return self._regex_chunk(path, language)

    # ------------------------------------------------------------------
    # Tree-sitter path
    # ------------------------------------------------------------------

    def _parse_with_treesitter(self, path: Path, ts_parser: object, language: str) -> List[RawChunk]:
        """Extract chunks using tree-sitter AST traversal."""
        source = path.read_bytes()
        tree = ts_parser.parse(source)  # type: ignore[union-attr]
        source_text = path.read_text(errors="replace")
        lines = source_text.splitlines()

        chunks: List[RawChunk] = []

        def _visit(node: object) -> None:
            node_type = getattr(node, "type", "")
            if node_type in ("function_definition", "class_definition"):
                start = getattr(node, "start_point", (0, 0))
                end = getattr(node, "end_point", (0, 0))
                start_line = start[0] + 1  # 1-based
                end_line = end[0] + 1

                content_lines = lines[start_line - 1 : end_line]
                content = "\n".join(content_lines)

                chunks.append(
                    RawChunk(
                        source_file=str(path),
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        language=language,
                    )
                )
            # Recurse into children
            for child in getattr(node, "children", []):
                _visit(child)

        _visit(tree.root_node)

        if not chunks:
            chunks = self._regex_chunk(path, language)

        return chunks

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    def _regex_chunk(self, path: Path, language: str) -> List[RawChunk]:
        """Robust regex-based chunker used when tree-sitter is unavailable."""
        try:
            source = path.read_text(errors="replace")
        except OSError:
            return []

        if not source.strip():
            # Empty file → one empty chunk so callers always get at least one.
            return [
                RawChunk(
                    source_file=str(path),
                    start_line=1,
                    end_line=1,
                    content="",
                    language=language,
                )
            ]

        # For Python (and similar indented languages) use declaration-aware splitting.
        if language == "python":
            return self._python_regex_chunks(path, source, language)

        # Generic: split into N-line windows.
        return self._line_split_chunks(path, source, language)

    def _python_regex_chunks(self, path: Path, source: str, language: str) -> List[RawChunk]:
        """Split a Python source string on top-level and class-level def/class lines."""
        lines = source.splitlines(keepends=True)
        total = len(lines)

        # Find all line indices (0-based) that start a new declaration block.
        boundary_indices: List[int] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                boundary_indices.append(i)

        if not boundary_indices:
            # No declarations found — return the whole file as one chunk.
            return [
                RawChunk(
                    source_file=str(path),
                    start_line=1,
                    end_line=total,
                    content=source,
                    language=language,
                )
            ]

        chunks: List[RawChunk] = []

        # If there's content before the first declaration, include it as a leading chunk
        # (e.g. module docstrings, imports).
        if boundary_indices[0] > 0:
            content = "".join(lines[: boundary_indices[0]])
            chunks.append(
                RawChunk(
                    source_file=str(path),
                    start_line=1,
                    end_line=boundary_indices[0],
                    content=content.rstrip(),
                    language=language,
                )
            )

        for idx, start_idx in enumerate(boundary_indices):
            end_idx = boundary_indices[idx + 1] if idx + 1 < len(boundary_indices) else total
            content = "".join(lines[start_idx:end_idx])
            chunks.append(
                RawChunk(
                    source_file=str(path),
                    start_line=start_idx + 1,  # 1-based
                    end_line=end_idx,  # inclusive of blank lines before next declaration
                    content=content.rstrip(),
                    language=language,
                )
            )

        return chunks

    def _line_split_chunks(self, path: Path, source: str, language: str) -> List[RawChunk]:
        """Split *source* into fixed-size line windows."""
        lines = source.splitlines(keepends=True)
        chunks: List[RawChunk] = []

        for offset in range(0, max(len(lines), 1), _FALLBACK_CHUNK_LINES):
            window = lines[offset : offset + _FALLBACK_CHUNK_LINES]
            if not window:
                break
            content = "".join(window).rstrip()
            start_line = offset + 1
            end_line = offset + len(window)
            chunks.append(
                RawChunk(
                    source_file=str(path),
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    language=language,
                )
            )

        return chunks or [
            RawChunk(
                source_file=str(path),
                start_line=1,
                end_line=1,
                content=source.rstrip(),
                language=language,
            )
        ]
