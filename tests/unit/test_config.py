"""Phase 1 Testing Gate — test_config.py

Tests typed access, defaults, TOML loading, partial overrides, and
validation for Config / ModelConfig / RoutingConfig / PathsConfig.
"""

from pathlib import Path

import pytest

from libucks.config import Config, ModelConfig, PathsConfig, RoutingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_toml(tmp_path: Path, content: str) -> Path:
    config_dir = tmp_path / ".libucks"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


# ---------------------------------------------------------------------------
# Defaults (no TOML file present)
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_load_missing_file_returns_config(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg, Config)

    def test_default_embedding_model(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.model.embedding_model == "all-MiniLM-L6-v2"

    def test_default_anthropic_model_is_set(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.model.anthropic_model, str)
        assert len(cfg.model.anthropic_model) > 0

    def test_default_novelty_threshold(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.routing.novelty_threshold == pytest.approx(0.35)

    def test_default_top_k(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.routing.top_k == 3

    def test_default_mitosis_threshold(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.routing.mitosis_threshold == 20_000

    def test_default_bucket_dir(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.paths.bucket_dir == ".libucks/buckets"

    def test_default_registry_file(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.paths.registry_file == ".libucks/registry.json"

    def test_default_pending_events(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.paths.pending_events == ".libucks/pending_events.jsonl"

    def test_default_log_file(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert cfg.paths.log_file == ".libucks/libucks.log"

    def test_direct_construction_equals_load_from_missing(self, tmp_path: Path):
        loaded = Config.load(tmp_path)
        direct = Config()
        assert loaded.routing.novelty_threshold == direct.routing.novelty_threshold
        assert loaded.routing.top_k == direct.routing.top_k
        assert loaded.routing.mitosis_threshold == direct.routing.mitosis_threshold
        assert loaded.model.embedding_model == direct.model.embedding_model
        assert loaded.paths.bucket_dir == direct.paths.bucket_dir


# ---------------------------------------------------------------------------
# Full TOML loading
# ---------------------------------------------------------------------------

class TestFullTomlLoading:
    def test_model_section_loaded(self, tmp_path: Path):
        _write_toml(tmp_path, """
[model]
anthropic_model = "claude-3-5-sonnet-20241022"
embedding_model = "paraphrase-MiniLM-L6-v2"
""")
        cfg = Config.load(tmp_path)
        assert cfg.model.anthropic_model == "claude-3-5-sonnet-20241022"
        assert cfg.model.embedding_model == "paraphrase-MiniLM-L6-v2"

    def test_routing_section_loaded(self, tmp_path: Path):
        _write_toml(tmp_path, """
[routing]
novelty_threshold = 0.5
top_k = 5
mitosis_threshold = 8000
""")
        cfg = Config.load(tmp_path)
        assert cfg.routing.novelty_threshold == pytest.approx(0.5)
        assert cfg.routing.top_k == 5
        assert cfg.routing.mitosis_threshold == 8000

    def test_paths_section_loaded(self, tmp_path: Path):
        _write_toml(tmp_path, """
[paths]
bucket_dir = "custom/buckets"
registry_file = "custom/registry.json"
log_file = "custom/app.log"
""")
        cfg = Config.load(tmp_path)
        assert cfg.paths.bucket_dir == "custom/buckets"
        assert cfg.paths.registry_file == "custom/registry.json"
        assert cfg.paths.log_file == "custom/app.log"

    def test_all_sections_together(self, tmp_path: Path):
        _write_toml(tmp_path, """
[model]
embedding_model = "custom-model"

[routing]
top_k = 7

[paths]
bucket_dir = "my/buckets"
""")
        cfg = Config.load(tmp_path)
        assert cfg.model.embedding_model == "custom-model"
        assert cfg.routing.top_k == 7
        assert cfg.paths.bucket_dir == "my/buckets"


# ---------------------------------------------------------------------------
# Partial TOML — missing sections and keys fall back to defaults
# ---------------------------------------------------------------------------

class TestPartialToml:
    def test_missing_routing_section_uses_defaults(self, tmp_path: Path):
        _write_toml(tmp_path, """
[model]
embedding_model = "custom-model"
""")
        cfg = Config.load(tmp_path)
        assert cfg.routing.novelty_threshold == pytest.approx(0.35)
        assert cfg.routing.top_k == 3
        assert cfg.routing.mitosis_threshold == 20_000

    def test_missing_model_section_uses_defaults(self, tmp_path: Path):
        _write_toml(tmp_path, """
[routing]
top_k = 5
""")
        cfg = Config.load(tmp_path)
        assert cfg.model.embedding_model == "all-MiniLM-L6-v2"

    def test_partial_routing_fills_missing_keys(self, tmp_path: Path):
        _write_toml(tmp_path, """
[routing]
top_k = 10
""")
        cfg = Config.load(tmp_path)
        assert cfg.routing.top_k == 10
        # Other routing keys must still be defaults
        assert cfg.routing.novelty_threshold == pytest.approx(0.35)
        assert cfg.routing.mitosis_threshold == 20_000

    def test_partial_paths_fills_missing_keys(self, tmp_path: Path):
        _write_toml(tmp_path, """
[paths]
bucket_dir = "special/buckets"
""")
        cfg = Config.load(tmp_path)
        assert cfg.paths.bucket_dir == "special/buckets"
        assert cfg.paths.registry_file == ".libucks/registry.json"
        assert cfg.paths.log_file == ".libucks/libucks.log"

    def test_empty_toml_file_uses_all_defaults(self, tmp_path: Path):
        _write_toml(tmp_path, "")
        cfg = Config.load(tmp_path)
        assert cfg.routing.top_k == 3
        assert cfg.model.embedding_model == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Type correctness
# ---------------------------------------------------------------------------

class TestTypes:
    def test_novelty_threshold_is_float(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.routing.novelty_threshold, float)

    def test_top_k_is_int(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.routing.top_k, int)

    def test_mitosis_threshold_is_int(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.routing.mitosis_threshold, int)

    def test_embedding_model_is_str(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.model.embedding_model, str)

    def test_bucket_dir_is_str(self, tmp_path: Path):
        cfg = Config.load(tmp_path)
        assert isinstance(cfg.paths.bucket_dir, str)

    def test_int_fields_coerced_from_float_on_construction(self):
        """top_k and mitosis_threshold are coerced to int in __post_init__,
        so passing floats (e.g. from a looser config source) is safe."""
        rc = RoutingConfig(novelty_threshold=0.5, top_k=3.0, mitosis_threshold=20_000.0)  # type: ignore[arg-type]
        assert type(rc.top_k) is int
        assert type(rc.mitosis_threshold) is int
        assert rc.top_k == 3
        assert rc.mitosis_threshold == 20_000


# ---------------------------------------------------------------------------
# Validation — out-of-range or invalid values raise ValueError
# ---------------------------------------------------------------------------

class TestValidation:
    def test_novelty_threshold_zero_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\nnovelty_threshold = 0.0\n")
        with pytest.raises(ValueError, match="novelty_threshold"):
            Config.load(tmp_path)

    def test_novelty_threshold_one_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\nnovelty_threshold = 1.0\n")
        with pytest.raises(ValueError, match="novelty_threshold"):
            Config.load(tmp_path)

    def test_novelty_threshold_negative_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\nnovelty_threshold = -0.1\n")
        with pytest.raises(ValueError, match="novelty_threshold"):
            Config.load(tmp_path)

    def test_top_k_zero_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\ntop_k = 0\n")
        with pytest.raises(ValueError, match="top_k"):
            Config.load(tmp_path)

    def test_top_k_negative_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\ntop_k = -1\n")
        with pytest.raises(ValueError, match="top_k"):
            Config.load(tmp_path)

    def test_mitosis_threshold_zero_raises(self, tmp_path: Path):
        _write_toml(tmp_path, "[routing]\nmitosis_threshold = 0\n")
        with pytest.raises(ValueError, match="mitosis_threshold"):
            Config.load(tmp_path)

    def test_valid_boundary_novelty_threshold(self, tmp_path: Path):
        """Values strictly between 0 and 1 must not raise."""
        _write_toml(tmp_path, "[routing]\nnovelty_threshold = 0.01\n")
        cfg = Config.load(tmp_path)
        assert cfg.routing.novelty_threshold == pytest.approx(0.01)

    def test_direct_construction_validates_too(self):
        with pytest.raises(ValueError, match="novelty_threshold"):
            RoutingConfig(novelty_threshold=0.0)

    def test_direct_construction_valid_values_ok(self):
        rc = RoutingConfig(novelty_threshold=0.5, top_k=5, mitosis_threshold=10_000)
        assert rc.top_k == 5
