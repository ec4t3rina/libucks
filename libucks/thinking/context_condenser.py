"""ContextCondenser — extracts a token-budget-safe digest of source code.

Priority order (greedy fill):
  1. Module-level docstring (file overview)
  2. Function / class signatures + their docstrings
  3. Hard truncation at the token boundary

Only signatures and docstrings are retained when the budget is tight.
Implementation bodies are discarded unless budget permits including them.
No model or torch dependency — pure Python / AST.
"""
from __future__ import annotations

import ast
from typing import List

from libucks.models.chunk import RawChunk

# Must match init_orchestrator._TOKENS_PER_CHAR for consistency.
_TOKENS_PER_CHAR: float = 0.25


def _rough_tokens(text: str) -> int:
    return max(1, int(len(text) * _TOKENS_PER_CHAR))


def _truncate_to_budget(text: str, remaining_tokens: int) -> str:
    max_chars = int(remaining_tokens / _TOKENS_PER_CHAR)
    return text[:max_chars]


class ContextCondenser:
    """Produces a condensed, token-budget-constrained digest of RawChunk source code."""

    def condense(self, chunks: List[RawChunk], budget_tokens: int = 200) -> str:
        """Return a string representing the collective essence of *chunks*.

        Stays within *budget_tokens* as measured by _rough_tokens().
        """
        if not chunks:
            return ""

        parts: List[str] = []
        used = 0

        for chunk in chunks:
            if used >= budget_tokens:
                break

            if chunk.language == "python":
                skeleton = self._python_skeleton(chunk.content)
            else:
                skeleton = [chunk.content]

            for item in skeleton:
                # Each item costs its own tokens plus 1 token for the "\n" separator
                # that "\n".join() will insert between items.
                sep = 1 if parts else 0
                t = _rough_tokens(item) + sep
                remaining = budget_tokens - used
                if t <= remaining:
                    parts.append(item)
                    used += t
                else:
                    truncated = _truncate_to_budget(item, remaining - sep)
                    if truncated.strip():
                        parts.append(truncated)
                        used += _rough_tokens(truncated) + sep
                    # Budget exhausted — stop processing this chunk.
                    break

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _python_skeleton(self, content: str) -> List[str]:
        """Extract an ordered list of skeleton strings from Python source.

        Order:
          1. Module docstring (if present)
          2. Top-level and nested function/class signatures + docstrings
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return [content]

        result: List[str] = []

        module_doc = ast.get_docstring(tree)
        if module_doc:
            result.append(f'"""{module_doc}"""')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._func_signature(node)
                doc = ast.get_docstring(node)
                result.append(f"{sig}\n    \"\"\"{doc}\"\"\"" if doc else sig)
            elif isinstance(node, ast.ClassDef):
                sig = f"class {node.name}:"
                doc = ast.get_docstring(node)
                result.append(f"{sig}\n    \"\"\"{doc}\"\"\"" if doc else sig)

        return result or [content]

    def _func_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = self._format_args(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({args}){ret}:"

    @staticmethod
    def _format_args(args: ast.arguments) -> str:
        parts: List[str] = []
        for arg in args.args:
            parts.append(
                f"{arg.arg}: {ast.unparse(arg.annotation)}" if arg.annotation else arg.arg
            )
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        for arg in args.kwonlyargs:
            parts.append(arg.arg)
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")
        return ", ".join(parts)
