"""Tests for _load_repo_path() resolution tiers."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


def _call_load(env_var: str | None, active_repo_content: str | None, tmp_path: Path) -> Path:
    """Call _load_repo_path with controlled env and home directory."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    libucks_dir = fake_home / ".libucks"

    if active_repo_content is not None:
        libucks_dir.mkdir()
        (libucks_dir / "active_repo").write_text(active_repo_content)

    env = {**os.environ}
    if env_var is not None:
        env["LIBUCKS_REPO_PATH"] = env_var
    else:
        env.pop("LIBUCKS_REPO_PATH", None)

    with patch.dict(os.environ, env, clear=True), \
         patch("pathlib.Path.home", return_value=fake_home):
        from libucks.mcp_bridge import _load_repo_path
        return _load_repo_path()


def test_env_var_wins(tmp_path: Path) -> None:
    target = tmp_path / "explicit_repo"
    target.mkdir()
    result = _call_load(str(target), active_repo_content=str(tmp_path / "other"), tmp_path=tmp_path)
    assert result == target.resolve()


def test_active_repo_used_when_no_env_var(tmp_path: Path) -> None:
    target = tmp_path / "active_repo_dir"
    target.mkdir()
    result = _call_load(env_var=None, active_repo_content=str(target), tmp_path=tmp_path)
    assert result == target.resolve()


def test_fallback_when_neither_set(tmp_path: Path) -> None:
    result = _call_load(env_var=None, active_repo_content=None, tmp_path=tmp_path)
    # Fallback is __file__.parent.parent — the libucks project root.
    # Just assert it's a valid directory, not the filesystem root.
    assert result.is_dir()
    assert result != Path("/")


def test_empty_active_repo_file_falls_through(tmp_path: Path) -> None:
    result = _call_load(env_var=None, active_repo_content="   ", tmp_path=tmp_path)
    assert result.is_dir()
    assert result != Path("/")
