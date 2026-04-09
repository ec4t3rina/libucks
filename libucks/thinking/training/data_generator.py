"""MultiPerspectiveDataGenerator — produces contrastive training triplets.

For each bucket, the V1 TextStrategy teacher generates three complementary
perspectives on the bucket's prose:
  1. Summary      — what the code does, concisely
  2. Logic Flow   — control structure and algorithmic decisions
  3. Dependency   — what this code depends on and what depends on it

Each perspective is encoded by the LatentStrategy into a hidden-state tensor.
These three tensors form the "Multi-Perspective" positive group.

Hard Negatives are found by scanning centroids for buckets that are topically
adjacent (cosine similarity in [NEG_SIM_LO, NEG_SIM_HI]) — close enough to
be confusing, but representing a different domain.

The teacher's summary text is also encoded to form the target latent that
the CommunicationAdapter should learn to align with.
"""
from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _read_chunk_content(meta) -> str:
    """Read lines [start_line, end_line] from the chunk's source file."""
    try:
        lines = Path(meta.source_file).read_text(errors="replace").splitlines()
        return "\n".join(lines[meta.start_line - 1 : meta.end_line])
    except OSError:
        return ""


def _collect_source_text(front_matter, max_chars: int = 3000) -> str:
    """Concatenate actual code content from ChunkMetadata, up to max_chars."""
    if not hasattr(front_matter, "chunks"):
        return ""
    parts = []
    total = 0
    for meta in front_matter.chunks:
        content = _read_chunk_content(meta)
        if not content:
            continue
        block = f"# {meta.source_file}\n{content}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n\n".join(parts)

# ---------------------------------------------------------------------------
# Three canonical perspectives used for every training sample.
# Tests assert that exactly these prompts are used.
# ---------------------------------------------------------------------------
PERSPECTIVE_PROMPTS: tuple[str, str, str] = (
    "Explain concisely what this code does and how it works.",
    "Describe the key algorithmic decisions and control flow in this code.",
    "What does this code depend on, and what other components use it?",
)

# Cosine-similarity window that defines "topically close but not the answer"
NEG_SIM_LO: float = 0.25
NEG_SIM_HI: float = 0.65
MAX_NEGATIVES: int = 3


@dataclasses.dataclass
class TrainingSample:
    """One contrastive training triplet for the CommunicationAdapter."""

    librarian_latents: List[torch.Tensor]
    """Three hidden-state tensors — one per perspective — from LatentStrategy.encode()."""

    target_latent: torch.Tensor
    """Hidden-state encoding of the teacher's summary text (the learning target)."""

    hard_negatives: List[torch.Tensor]
    """Hidden-state encodings of topically-adjacent but wrong bucket prose."""

    target_text: str
    """The teacher's plain-text summary (for debugging / manual review)."""


class MultiPerspectiveDataGenerator:
    """Generates TrainingSamples by combining a V1 teacher with a V2 LatentStrategy.

    Args:
        text_strategy:   V1 TextStrategy (calls Anthropic API) — the teacher.
        latent_strategy: V2 LatentStrategy (local model) — produces tensors.
        registry:        BucketRegistry — for centroid-based negative mining.
        store:           BucketStore — for reading bucket prose.
    """

    def __init__(
        self,
        text_strategy,
        latent_strategy,
        registry,
        store,
    ) -> None:
        self._text_strategy = text_strategy
        self._latent_strategy = latent_strategy
        self._registry = registry
        self._store = store

    async def generate(self, bucket_id: str) -> TrainingSample:
        """Generate a TrainingSample for the given bucket.

        Args:
            bucket_id: The bucket to build a training sample for.

        Returns:
            A TrainingSample with 3 perspective latents, a target latent,
            hard negatives, and the teacher's summary text.
        """
        _, prose = self._store.read(bucket_id)

        # ------------------------------------------------------------------ #
        # Step 1: Three-perspective encoding
        # ------------------------------------------------------------------ #
        teacher_texts: list[str] = []
        for prompt in PERSPECTIVE_PROMPTS:
            text = await self._text_strategy.reason(prompt, prose)
            teacher_texts.append(text)

        librarian_latents: list[torch.Tensor] = []
        for text in teacher_texts:
            latent = await self._latent_strategy.encode(text)
            librarian_latents.append(latent)

        # ------------------------------------------------------------------ #
        # Step 2: Target latent — encode the summary perspective
        # ------------------------------------------------------------------ #
        summary_text = teacher_texts[0]
        target_latent: torch.Tensor = await self._latent_strategy.encode(summary_text)

        # ------------------------------------------------------------------ #
        # Step 3: Hard negative mining via centroid cosine similarity
        # ------------------------------------------------------------------ #
        hard_negatives = await self._mine_hard_negatives(bucket_id, prose)

        return TrainingSample(
            librarian_latents=librarian_latents,
            target_latent=target_latent,
            hard_negatives=hard_negatives,
            target_text=summary_text,
        )

    async def generate_curriculum_batch(
        self,
        bucket_id: str,
        adapter: Any,
        tokenizer: Any,
        embedding: Any,
        output_len: int = 32,
        hidden_dim: int = 2048,
    ) -> Dict[str, Any]:
        """Generate one curriculum training item for the LoRA receiver.

        Steps:
          1. Read bucket prose and run it through the adapter to get a soft-prompt (K, D).
          2. Tokenize the target text (teacher summary) to get integer target IDs.
          3. Sample r ~ U[0, 1] and produce token embeddings for the plan prefix.
          4. Return CurriculumMixer.mix(soft_prompt, tok_embeds, r) as mixed_input.

        Args:
            bucket_id:  Bucket to generate training data for.
            adapter:    CommunicationAdapter — forward(representations) → (K, D).
            tokenizer:  Base model tokenizer — used to tokenize target text.
            embedding:  Base model embedding layer — maps token IDs to (K, D) embeds.
            output_len: K — number of soft-prompt tokens.
            hidden_dim: D — hidden dimension.

        Returns:
            Dict with keys:
              - 'mixed_input': Tensor (K, D)
              - 'target_ids':  LongTensor of target token IDs
              - 'r':           float in [0, 1]
        """
        from libucks.thinking.curriculum import CurriculumMixer

        front_matter, prose = self._store.read(bucket_id)

        # Use actual code content from disk; fall back to prose if no chunks readable
        source_text = _collect_source_text(front_matter, max_chars=3000) or prose

        # Get teacher summary text from actual code (not metadata prose)
        summary_text = await self._text_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)

        # Encode via reason() to match inference-time encoding distribution
        # (_handle_query also calls strategy.reason(), not encode())
        latent = await self._latent_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
        representations = [latent]

        # Run through adapter to get soft-prompt (K, D)
        soft_prompt: torch.Tensor = adapter(representations)

        # Tokenize target text → integer IDs
        # Determine device from the embedding layer so tensors are placed correctly.
        embed_device = next(embedding.parameters()).device
        enc = tokenizer(summary_text, return_tensors="pt")
        if isinstance(enc, dict) and "input_ids" in enc:
            target_ids = enc["input_ids"].squeeze(0).long().to(embed_device)
        else:
            target_ids = torch.tensor(
                tokenizer.encode(summary_text), dtype=torch.long, device=embed_device
            )

        # Get token embeddings for the plan (truncate/pad to output_len)
        plan_ids = target_ids[:output_len]
        if plan_ids.shape[0] < output_len:
            pad = torch.zeros(
                output_len - plan_ids.shape[0], dtype=torch.long, device=embed_device
            )
            plan_ids = torch.cat([plan_ids, pad])
        tok_embeds: torch.Tensor = embedding(plan_ids)   # (K, D)

        # Sample mixing rate and blend
        r = random.uniform(0.0, 1.0)
        mixed_input = CurriculumMixer.mix(soft_prompt, tok_embeds, r)

        return {
            "mixed_input": mixed_input,
            "target_ids": target_ids,
            "r": r,
        }

    async def _mine_hard_negatives(
        self, anchor_bucket_id: str, anchor_prose: str
    ) -> list[torch.Tensor]:
        """Find topically-adjacent buckets and encode their prose as negatives.

        Selects buckets whose centroid cosine similarity to the anchor falls
        in [NEG_SIM_LO, NEG_SIM_HI] — 'close enough to be confusing, different
        enough to be wrong'.
        """
        all_centroids: dict = self._registry.get_all_centroids()
        negatives: list[torch.Tensor] = []

        # Retrieve anchor centroid for comparison
        anchor_centroid = all_centroids.get(anchor_bucket_id)
        if anchor_centroid is None:
            return negatives

        anchor_vec = anchor_centroid.astype(np.float32)
        anchor_norm = anchor_vec / (np.linalg.norm(anchor_vec) + 1e-8)

        for other_id, other_centroid in all_centroids.items():
            if other_id == anchor_bucket_id:
                continue
            if len(negatives) >= MAX_NEGATIVES:
                break

            other_vec = other_centroid.astype(np.float32)
            other_norm = other_vec / (np.linalg.norm(other_vec) + 1e-8)
            sim = float(np.dot(anchor_norm, other_norm))

            if NEG_SIM_LO <= sim <= NEG_SIM_HI:
                _, other_prose = self._store.read(other_id)
                neg_latent = await self._latent_strategy.encode(other_prose)
                negatives.append(neg_latent)

        return negatives
