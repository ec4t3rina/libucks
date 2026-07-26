"""Config — typed configuration dataclasses loaded from .libucks/config.toml.

Uses Python 3.11's built-in tomllib (read-only, binary-mode).
All sub-dataclasses validate their own fields in __post_init__ so that
invalid values are caught at construction time regardless of whether
the Config came from a TOML file or was built directly in code.

Defaults are chosen to be sensible for a mid-sized repository:
  novelty_threshold  0.35   — cosine distance; tighter = fewer new buckets
  top_k              3      — buckets queried per request
  mitosis_threshold  20000  — tokens before a bucket splits
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """AI model identifiers."""

    anthropic_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "all-MiniLM-L6-v2"
    local_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    base_model: str = "Qwen/Qwen2.5-0.5B"
    quantization: str = "none"
    bnb_4bit_compute_dtype: str = "float32"
    device: str = "auto"
    strategy: str = "latent"
    compression_steps: int = 8
    """INERT — currently has no effect on any code path.

    `LatentCompressor` (thinking/compressor.py) is fully implemented and
    trainable via `ContrastiveAdapterTrainer.train_compressor_step`, and
    `LatentStrategy` will use one if given (`latent_strategy.py:218`) — but
    nothing in production ever constructs one. `thinking/__init__.py:20` builds
    `LatentStrategy(mgr)` with no compressor, so this value is read by nobody.

    Do NOT "fix" this by wiring a fresh LatentCompressor in: its query vectors
    are random at init, and injecting an untrained cross-attention bottleneck
    into the live latent path would corrupt every Representation. It would need
    a trained, persisted checkpoint first — there is no save/load path for one.

    Kept because the Phase 11 module and its tests are intact and the feature
    may be revived. Documented as dormant rather than deleted."""

    base_model_dtype: str = "float16"
    """Storage dtype for the Base receiver on MPS. float16 is fine for ~0.5B
    models; bigger models (1.5B+) may overflow fp16 and produce NaN losses
    during LoRA training — set to 'bfloat16' for those. Options: float16,
    bfloat16, float32. CUDA/CPU paths ignore this setting (transformers
    auto-picks fp32 there)."""

    output_len: int = 32
    """K — number of soft-prompt tokens the CommunicationAdapter emits per
    query. Changing this requires retraining adapter.pt and lora_receiver.pt
    (the output_queries parameter and downstream LoRA shapes are K-dependent).
    Larger K gives the latent path more capacity but quadratically scales
    inter-slot attention cost. 32 is the historical default; 64 trades
    compute for compression headroom on bigger receivers."""

    hybrid_retrieval: bool = False
    """Enable the verbatim retrieval channel alongside the latent channel.
    When True, Translator.synthesize fetches source text from each routed
    bucket and concatenates the embeddings before the soft prompt at decode
    time. Latent path still does cross-bucket fusion via the
    CommunicationAdapter; verbatim grounds identifier-level facts.

    Default False: libucks runs on any repo with latent-only by default;
    hybrid is per-repo opt-in via .libucks/config.toml."""

    hybrid_verbatim_max_chars: int = 3000
    """Total character budget for the verbatim channel, divided evenly across
    the top-k routed buckets. 3000 ≈ 750 tokens — under 5% of a 32K context
    window."""

    def __post_init__(self) -> None:
        if self.strategy != "latent":
            raise ValueError(
                f"strategy must be 'latent', got {self.strategy!r}"
            )
        if self.quantization not in ("none", "4bit", "8bit"):
            raise ValueError(
                f"quantization must be 'none', '4bit', or '8bit', "
                f"got {self.quantization!r}"
            )


@dataclass
class RoutingConfig:
    """Embedding-based routing parameters."""

    novelty_threshold: float = 0.35
    """Cosine distance.  A diff embedding farther than this from all
    existing centroids triggers CreateBucketEvent.  Range: (0, 1)."""

    top_k: int = 3
    """Number of buckets consulted per query.  Must be >= 1."""

    mitosis_threshold: int = 20_000
    """Token count at which a bucket is eligible for manual mitosis."""

    init_bucket_size: int = 2_000
    """Target raw-token count per bucket during INIT clustering.
    Controls how many buckets are seeded: n_clusters = total_tokens // init_bucket_size.
    Kept separate from mitosis_threshold so INIT density can be tuned independently
    of runtime splitting behaviour."""

    min_bucket_seed_tokens: int = 1_500
    """Minimum estimated token count required to spawn a new bucket from novel
    runtime content. Smaller diffs are routed to the nearest existing bucket
    even when cosine-distant. Buckets are meant to represent solid subjects;
    one-file spawns would fragment the index. HealthMonitor's coherence-driven
    mitosis is the natural pressure valve once enough material has accumulated
    in an existing bucket."""

    def __post_init__(self) -> None:
        self.novelty_threshold = float(self.novelty_threshold)
        self.top_k = int(self.top_k)
        self.mitosis_threshold = int(self.mitosis_threshold)
        self.init_bucket_size = int(self.init_bucket_size)
        self.min_bucket_seed_tokens = int(self.min_bucket_seed_tokens)

        if not (0.0 < self.novelty_threshold < 1.0):
            raise ValueError(
                f"novelty_threshold must be in the open interval (0, 1), "
                f"got {self.novelty_threshold}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.mitosis_threshold < 1:
            raise ValueError(
                f"mitosis_threshold must be >= 1, got {self.mitosis_threshold}"
            )
        if self.init_bucket_size < 1:
            raise ValueError(
                f"init_bucket_size must be >= 1, got {self.init_bucket_size}"
            )
        if self.min_bucket_seed_tokens < 1:
            raise ValueError(
                f"min_bucket_seed_tokens must be >= 1, got {self.min_bucket_seed_tokens}"
            )


@dataclass
class PathsConfig:
    """File-system paths used by libucks (relative to the target repo root
    unless prefixed with ~/)."""

    bucket_dir: str = ".libucks/buckets"
    registry_file: str = ".libucks/registry.json"
    pending_events: str = ".libucks/pending_events.jsonl"
    log_file: str = ".libucks/libucks.log"
    grammar_cache: str = "~/.libucks/grammars"
    repo_cache: str = "~/.libucks/repos"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

def _merge(cls: type, data: dict[str, Any]) -> object:
    """Construct a dataclass from a dict, ignoring unknown keys and letting
    unrecognised fields fall back to their declared defaults."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, repo_path: Path) -> "Config":
        """Load config from <repo_path>/.libucks/config.toml.

        Returns a Config with all defaults if the file does not exist.
        Sections or keys absent from the file are filled in with defaults.
        """
        config_file = repo_path / ".libucks" / "config.toml"
        if not config_file.exists():
            return cls()

        with open(config_file, "rb") as fh:
            data = tomllib.load(fh)

        return cls(
            model=_merge(ModelConfig, data.get("model", {})),       # type: ignore[arg-type]
            routing=_merge(RoutingConfig, data.get("routing", {})),  # type: ignore[arg-type]
            paths=_merge(PathsConfig, data.get("paths", {})),        # type: ignore[arg-type]
        )
