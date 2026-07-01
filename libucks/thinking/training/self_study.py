"""Self-study query generation (Cartridge Memory, CM-A).

The Cartridges recipe (arXiv 2506.06266) distills a corpus into a latent by
training on *self-generated* synthetic queries about it — hundreds per unit,
vs. libucks's historical ~3. This module produces those queries for one bucket,
locally ($0): the frozen instruct model proposes questions about the bucket's
own source, backed by a deterministic template fallback so we always reach the
requested count even if the model's list is short or malformed.

Answers are NOT produced here — the teacher (full-context frozen receiver)
generates them at distillation time. This module only supplies the *queries*.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Optional

import torch


def _log(msg: str) -> None:
    print(f"[libucks:self_study] {msg}", file=sys.stderr, flush=True)


_TEMPLATES = (
    "What does {tok} do in this code?",
    "How is {tok} used?",
    "What is the purpose of {tok}?",
    "Explain the role of {tok}.",
    "What are the inputs and outputs of {tok}?",
    "Which other components depend on {tok}?",
    "What happens when {tok} is called?",
    "Describe the control flow around {tok}.",
    "What edge cases does {tok} handle?",
    "What would break if {tok} were removed?",
)

# Identifier-ish tokens: function/class/method names, CONSTANTS, snake_case.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# All lowercase; membership is tested against `w.lower()`.
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "import", "return",
    "self", "none", "true", "false", "def", "class", "not", "are", "was",
    "int", "str", "list", "dict", "type", "value",
}


def _extract_identifiers(text: str, limit: int = 40) -> list[str]:
    seen: dict[str, int] = {}
    for m in _IDENT_RE.finditer(text):
        w = m.group(0)
        if w.lower() in _STOPWORDS:
            continue
        seen[w] = seen.get(w, 0) + 1
    # Prefer identifiers that recur (more central to the bucket).
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


def _template_queries(bucket_text: str, n: int) -> list[str]:
    idents = _extract_identifiers(bucket_text) or ["this module"]
    out: list[str] = []
    i = 0
    while len(out) < n:
        tok = idents[i % len(idents)]
        tmpl = _TEMPLATES[(i // len(idents)) % len(_TEMPLATES)]
        out.append(tmpl.format(tok=tok))
        i += 1
        if i > n * len(_TEMPLATES) * 2:  # safety valve
            break
    return out[:n]


@torch.no_grad()
def _model_queries(
    bucket_text: str,
    n: int,
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 400,
    max_context_chars: int = 3000,
) -> list[str]:
    """Ask an instruct model to propose `n` questions about the bucket source.

    Best-effort: returns whatever parses cleanly (may be < n; caller tops up
    with templates). Uses the chat template when available.
    """
    device = next(model.parameters()).device
    ctx = bucket_text[:max_context_chars]
    instruction = (
        f"Read the following source code and write {n} distinct, specific "
        "questions whose answers require facts from the code (function names, "
        "constants, control flow, defaults). One question per line, no numbering.\n\n"
        f"```\n{ctx}\n```\n\nQuestions:"
    )
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content": instruction}]
            input_ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        else:
            input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    except Exception as exc:  # model/generation hiccup → fall back to templates
        _log(f"model query-gen failed ({exc}); using templates only")
        return []

    queries: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line)  # strip bullets/numbering
        if len(line) < 8 or "?" not in line:
            continue
        queries.append(line)
    return queries


def generate_self_study_queries(
    bucket_text: str,
    n: int,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> list[str]:
    """Return exactly `n` self-study queries for a bucket.

    Uses the model if provided (deduped), then tops up to `n` with deterministic
    template queries. Always returns `n` items (assuming non-empty bucket_text).
    """
    queries: list[str] = []
    seen: set[str] = set()

    if model is not None and tokenizer is not None:
        for q in _model_queries(bucket_text, n, model, tokenizer):
            key = q.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q)
                if len(queries) >= n:
                    break

    if len(queries) < n:
        for q in _template_queries(bucket_text, n * 2):
            key = q.lower()
            if key not in seen:
                seen.add(key)
                queries.append(q)
                if len(queries) >= n:
                    break

    _log(f"generated {len(queries)} queries (target {n})")
    return queries[:n]
