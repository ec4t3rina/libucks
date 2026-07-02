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


# Fact-probing templates: phrased to force the full-context teacher to state
# SPECIFIC values (numbers, probabilities, thresholds, state names, defaults) —
# the identifiers the latent must carry. Generic "what does X do" templates
# (CM-A.1) let the teacher answer structurally without stating the facts, so the
# cartridge never learned them. See docs/cartridges-log.md CM-A.1.
_TEMPLATES = (
    "What is the exact numeric value, probability, or threshold associated with {tok}?",
    "State the precise default or constant used for {tok}.",
    "What specific state or condition does {tok} correspond to, exactly?",
    "Give the exact behavior of {tok}, including any numbers or probabilities.",
    "Under what precise conditions (with specific counts or values) does {tok} apply?",
    "What exact value does {tok} take, and what does it control?",
    "What does {tok} do in this code, with the specific values involved?",
    "How is {tok} used, and what are the exact parameters or thresholds?",
    "What happens, step by step with concrete values, when {tok} is triggered?",
    "Which specific constants, states, or probabilities are involved in {tok}?",
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
    per_call_new_tokens: int = 384,
    max_context_chars: int = 3000,
    max_attempts: Optional[int] = None,
) -> list[str]:
    """Ask an instruct model to propose fact-probing questions about the bucket.

    Loops `generate` (sampled, so each call yields different questions) until it
    has `n` distinct questions or exhausts `max_attempts`. Best-effort: caller
    tops up any shortfall with templates. Uses the chat template when available.
    """
    device = next(model.parameters()).device
    ctx = bucket_text[:max_context_chars]
    batch = max(8, min(30, n))  # ask for a chunk at a time
    instruction = (
        f"Read the following source code and write {batch} distinct, specific "
        "questions. Each answer MUST require a concrete fact from the code — an "
        "exact number, probability, threshold, default, constant, state name, or "
        "function name. Avoid vague questions. One question per line, no numbering.\n\n"
        f"```\n{ctx}\n```\n\nQuestions:"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = [{"role": "user", "content": instruction}]
        # return_dict=True → BatchEncoding with input_ids + attention_mask.
        # (Without it, this transformers version returns a BatchEncoding whose
        # .shape access raises, or a bare tensor — normalise both here.)
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    else:
        enc = tokenizer(instruction, return_tensors="pt")
    base_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    prompt_len = base_ids.shape[1]

    attempts = max_attempts if max_attempts is not None else min(30, 2 * ((n // batch) + 2))
    seen: set[str] = set()
    queries: list[str] = []
    for _ in range(attempts):
        if len(queries) >= n:
            break
        try:
            out = model.generate(
                base_ids,
                attention_mask=attn,
                max_new_tokens=per_call_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        except Exception as exc:
            _log(f"model query-gen failed ({exc}); stopping model gen")
            break
        for line in text.splitlines():
            line = re.sub(r"^\s*[-*\d.)\]]+\s*", "", line.strip())
            if len(line) < 8 or "?" not in line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(line)
    _log(f"model query-gen: {len(queries)} distinct questions in <= {attempts} attempts")
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
