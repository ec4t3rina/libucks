"""Unit tests for libucks.git_hook_receiver.

Tests:
  - install_hooks: creates new hook files with shebang + trigger line
  - install_hooks: appends to existing hook files without overwriting
  - install_hooks: idempotent (does not duplicate the trigger line)
  - serve_socket / IPC: payload sent over the Unix socket is dispatched to on_event
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import tempfile
from pathlib import Path

import pytest

from libucks.git_hook_receiver import _HOOK_EVENTS, _HOOK_LINE, find_git_root, install_hooks, serve_socket


@pytest.fixture()
def short_sock_path() -> Path:
    """Return a short Unix socket path that fits within macOS's 104-char AF_UNIX limit."""
    # Use /tmp with a short unique suffix — pytest's tmp_path is often too long.
    with tempfile.NamedTemporaryFile(suffix=".sock", dir="/tmp", delete=True) as f:
        p = Path(f.name)
    # File was deleted; return the path so server can create it
    return p


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_BIN = "/fake/venv/bin/libucks"


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """A minimal directory tree that looks like a git repo (.git/hooks/)."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# install_hooks — filesystem tests
# ---------------------------------------------------------------------------

class TestInstallHooks:
    def test_creates_new_hook_files(self, fake_repo: Path) -> None:
        modified, _ = install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)

        assert set(modified) == set(_HOOK_EVENTS)
        hooks_dir = fake_repo / ".git" / "hooks"
        for event in _HOOK_EVENTS:
            hook_file = hooks_dir / event
            assert hook_file.exists(), f"{event} hook not created"
            content = hook_file.read_text()
            assert content.startswith("#!/bin/sh\n"), f"{event} missing shebang"
            expected = _HOOK_LINE.format(libucks_path=fake_repo.resolve(), libucks_bin=FAKE_BIN, event=event)
            assert expected in content

    def test_new_hook_files_are_executable(self, fake_repo: Path) -> None:
        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)
        hooks_dir = fake_repo / ".git" / "hooks"
        for event in _HOOK_EVENTS:
            hook_file = hooks_dir / event
            mode = hook_file.stat().st_mode
            assert mode & stat.S_IXUSR, f"{event} hook is not executable"

    def test_appends_to_existing_hook(self, fake_repo: Path) -> None:
        hooks_dir = fake_repo / ".git" / "hooks"
        existing_content = "#!/bin/sh\necho 'existing hook'\n"
        hook_file = hooks_dir / "post-commit"
        hook_file.write_text(existing_content)

        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)

        result = hook_file.read_text()
        assert "echo 'existing hook'" in result, "existing content was erased"
        expected = _HOOK_LINE.format(libucks_path=fake_repo.resolve(), libucks_bin=FAKE_BIN, event="post-commit")
        assert expected in result, "trigger not appended"
        assert result.count("#!/bin/sh") == 1

    def test_idempotent_does_not_double_append(self, fake_repo: Path) -> None:
        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)
        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)  # second call should be a no-op

        hooks_dir = fake_repo / ".git" / "hooks"
        for event in _HOOK_EVENTS:
            content = (hooks_dir / event).read_text()
            trigger = _HOOK_LINE.format(libucks_path=fake_repo.resolve(), libucks_bin=FAKE_BIN, event=event)
            assert content.count(trigger) == 1, f"trigger duplicated for {event}"

    def test_returns_empty_list_when_already_installed(self, fake_repo: Path) -> None:
        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)
        second_pass, _ = install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)
        assert second_pass == [], "expected [] on no-op second install"

    def test_creates_hooks_dir_if_missing(self, tmp_path: Path) -> None:
        """install_hooks creates .git/hooks/ if it does not exist."""
        (tmp_path / ".git").mkdir()
        modified, _ = install_hooks(tmp_path, git_root=tmp_path, libucks_bin=FAKE_BIN)
        assert set(modified) == set(_HOOK_EVENTS)

    def test_subdirectory_installs_in_git_root(self, tmp_path: Path) -> None:
        """When libucks_path is a subdirectory, hooks land in the git root."""
        git_root = tmp_path / "repo"
        libucks_path = git_root / "src" / "mylib"
        (git_root / ".git" / "hooks").mkdir(parents=True)
        libucks_path.mkdir(parents=True)

        modified, _ = install_hooks(libucks_path, git_root=git_root, libucks_bin=FAKE_BIN)

        hooks_dir = git_root / ".git" / "hooks"
        assert set(modified) == set(_HOOK_EVENTS)
        for event in _HOOK_EVENTS:
            content = (hooks_dir / event).read_text()
            expected = _HOOK_LINE.format(libucks_path=libucks_path.resolve(), libucks_bin=FAKE_BIN, event=event)
            assert expected in content, f"hook for {event} missing LIBUCKS_REPO_PATH"
            assert str(libucks_path.resolve()) in content

    def test_absolute_binary_path_written_to_hook(self, fake_repo: Path) -> None:
        """The hook script must contain the absolute binary path, not 'libucks'."""
        install_hooks(fake_repo, git_root=fake_repo, libucks_bin=FAKE_BIN)
        content = (fake_repo / ".git" / "hooks" / "post-commit").read_text()
        assert FAKE_BIN in content
        assert content.count("libucks hook") == 1  # no bare 'libucks' on its own line

    def test_force_replaces_stale_trigger(self, fake_repo: Path) -> None:
        """--force removes old libucks lines and writes the current trigger."""
        hooks_dir = fake_repo / ".git" / "hooks"
        stale = "#!/bin/sh\nlibucks hook post-commit \"$@\" || true\n"
        (hooks_dir / "post-commit").write_text(stale)

        modified, _ = install_hooks(fake_repo, git_root=fake_repo, force=True, libucks_bin=FAKE_BIN)

        content = (hooks_dir / "post-commit").read_text()
        assert FAKE_BIN in content
        fresh_trigger = _HOOK_LINE.format(libucks_path=fake_repo.resolve(), libucks_bin=FAKE_BIN, event="post-commit")
        assert fresh_trigger in content
        assert content.count("libucks hook post-commit") == 1

    def test_find_git_root_walks_up(self, tmp_path: Path) -> None:
        """find_git_root should return the ancestor that contains .git/."""
        git_root = tmp_path / "repo"
        sub = git_root / "src" / "pkg"
        (git_root / ".git").mkdir(parents=True)
        sub.mkdir(parents=True)

        assert find_git_root(sub) == git_root

    def test_find_git_root_raises_outside_repo(self, tmp_path: Path) -> None:
        """find_git_root should raise RuntimeError if no .git found."""
        with pytest.raises(RuntimeError, match="No .git"):
            find_git_root(tmp_path)


# ---------------------------------------------------------------------------
# serve_socket / IPC — real Unix socket, async
# ---------------------------------------------------------------------------

class TestServeSocket:
    async def _send_payload(self, sock_path: Path, payload: dict) -> None:
        """Helper: open the socket, send JSON, close."""
        # Wait until the server has bound a real Unix socket (not just any file).
        for _ in range(40):
            if sock_path.exists() and stat.S_ISSOCK(sock_path.stat().st_mode):
                break
            await asyncio.sleep(0.05)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: _sync_send(sock_path, payload),
        )

    @pytest.mark.asyncio
    async def test_dispatches_event_to_on_event(self, short_sock_path: Path) -> None:
        received: list[dict] = []

        async def on_event(payload: dict) -> None:
            received.append(payload)

        server_task = asyncio.ensure_future(serve_socket(short_sock_path, on_event))
        try:
            await self._send_payload(short_sock_path, {"event": "post-commit"})
            # Give the server a moment to invoke on_event
            await asyncio.sleep(0.1)
            assert received == [{"event": "post-commit"}]
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_handles_multiple_connections(self, short_sock_path: Path) -> None:
        received: list[dict] = []

        async def on_event(payload: dict) -> None:
            received.append(payload)

        server_task = asyncio.ensure_future(serve_socket(short_sock_path, on_event))
        try:
            for event_name in ["post-commit", "post-checkout", "post-rewrite"]:
                await self._send_payload(short_sock_path, {"event": event_name})
                await asyncio.sleep(0.05)

            await asyncio.sleep(0.1)
            events = [p["event"] for p in received]
            assert "post-commit" in events
            assert "post-checkout" in events
            assert "post-rewrite" in events
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_removes_stale_socket_on_start(self, short_sock_path: Path) -> None:
        # Create a stale file at the socket path
        short_sock_path.write_bytes(b"stale")

        received: list[dict] = []

        async def on_event(payload: dict) -> None:
            received.append(payload)

        server_task = asyncio.ensure_future(serve_socket(short_sock_path, on_event))
        try:
            await self._send_payload(short_sock_path, {"event": "post-commit"})
            await asyncio.sleep(0.1)
            assert len(received) == 1
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_ignores_malformed_json(self, short_sock_path: Path) -> None:
        """Malformed JSON must not crash the server."""
        received: list[dict] = []

        async def on_event(payload: dict) -> None:
            received.append(payload)

        server_task = asyncio.ensure_future(serve_socket(short_sock_path, on_event))
        try:
            # Wait for socket to be bound (not just any file)
            for _ in range(40):
                if short_sock_path.exists() and stat.S_ISSOCK(short_sock_path.stat().st_mode):
                    break
                await asyncio.sleep(0.05)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: _sync_send_raw(short_sock_path, b"not json!!"))
            await asyncio.sleep(0.1)

            # Server still alive — send a valid event
            await self._send_payload(short_sock_path, {"event": "post-commit"})
            await asyncio.sleep(0.1)
            assert received == [{"event": "post-commit"}]
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Sync helpers (run in executor to avoid blocking the event loop)
# ---------------------------------------------------------------------------

def _sync_send(sock_path: Path, payload: dict) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        s.connect(str(sock_path))
        s.sendall(json.dumps(payload).encode())


def _sync_send_raw(sock_path: Path, data: bytes) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(3)
        s.connect(str(sock_path))
        s.sendall(data)
