"""Tests for ContextCondenser — signature/docstring digest within a token budget."""
from __future__ import annotations

import textwrap

import pytest

from libucks.models.chunk import RawChunk

_TOKENS_PER_CHAR = 0.25


def _rough_tokens(text: str) -> int:
    return max(1, int(len(text) * _TOKENS_PER_CHAR))


def _make_chunk(source_file: str, content: str, language: str = "python") -> RawChunk:
    lines = content.splitlines()
    return RawChunk(
        source_file=source_file,
        start_line=1,
        end_line=max(1, len(lines)),
        content=content,
        language=language,
    )


class TestContextCondenserBudget:
    def test_output_fits_within_budget(self):
        content = "\n".join(
            [f"def func_{i}(x, y):\n    return x + y + {i}" for i in range(50)]
        )
        chunk = _make_chunk("/repo/module.py", content)

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk], budget_tokens=100)

        assert _rough_tokens(result) <= 100

    def test_large_budget_includes_more_content(self):
        content = "\n\n".join([
            f"def func_{i}(x, y):\n    '''Does thing {i}.'''\n    return x + y"
            for i in range(20)
        ])
        chunk = _make_chunk("/repo/module.py", content)

        from libucks.thinking.context_condenser import ContextCondenser
        condenser = ContextCondenser()
        small = condenser.condense([chunk], budget_tokens=50)
        large = condenser.condense([chunk], budget_tokens=500)

        assert len(large) >= len(small)

    def test_empty_chunks_returns_empty_string(self):
        from libucks.thinking.context_condenser import ContextCondenser
        assert ContextCondenser().condense([], budget_tokens=200) == ""


class TestContextCondenserContent:
    def test_function_signatures_are_present(self):
        content = textwrap.dedent("""\
            def process_payment(amount: float, currency: str) -> bool:
                \"\"\"Process a payment transaction.\"\"\"
                internal = amount * 1.1
                return True
        """)
        chunk = _make_chunk("/repo/payments.py", content)

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk], budget_tokens=200)

        assert "process_payment" in result

    def test_docstrings_are_present(self):
        content = textwrap.dedent("""\
            def authenticate(token: str) -> bool:
                \"\"\"Validate a JWT token and return True if valid.\"\"\"
                return True
        """)
        chunk = _make_chunk("/repo/auth.py", content)

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk], budget_tokens=200)

        assert "Validate a JWT token" in result

    def test_module_docstring_appears_in_output(self):
        content = textwrap.dedent("""\
            \"\"\"Authentication module for handling JWT tokens.\"\"\"

            def authenticate(token: str) -> bool:
                return True
        """)
        chunk = _make_chunk("/repo/auth.py", content)

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk], budget_tokens=200)

        assert "Authentication module" in result

    def test_signatures_present_even_when_bodies_trimmed(self):
        """At tight budget, signatures should appear even if bodies are dropped."""
        functions = "\n\n".join([
            f"def func_{i}(a, b, c):\n    '''Docstring for func {i}.'''\n    x = a + b\n    return x + c"
            for i in range(10)
        ])
        chunk = _make_chunk("/repo/module.py", functions)

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk], budget_tokens=80)

        assert "def func_" in result

    def test_multiple_chunks_from_different_files(self):
        chunk_a = _make_chunk("/repo/a.py", "def foo(): pass")
        chunk_b = _make_chunk("/repo/b.py", "def bar(): pass")

        from libucks.thinking.context_condenser import ContextCondenser
        result = ContextCondenser().condense([chunk_a, chunk_b], budget_tokens=200)

        assert "foo" in result
        assert "bar" in result
