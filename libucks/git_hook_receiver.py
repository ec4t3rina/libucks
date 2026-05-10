"""GitHookReceiver — Unix domain socket listener for git hook events.

Git hook scripts call `libucks hook <event> "$@" || true` which sends a
single JSON line over the Unix socket at `.libucks/server.sock` and exits.

Supported payload shapes:
  {"event": "post-commit"}
  {"event": "post-checkout", "args": ["<prev_head>", "<new_head>", "1"]}
  {"event": "post-rewrite", "args": ["rebase"]}

The server reads the payload, calls *on_event*, and closes the connection.
Hook scripts never wait for a response — they fire-and-forget.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

# Type alias for the callback injected by mcp_bridge.
OnEventFn = Callable[[dict], Awaitable[None]]


async def _handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    on_event: OnEventFn,
) -> None:
    """Read one JSON payload, dispatch, close."""
    try:
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        payload: dict = json.loads(data.decode())
        log.info("git_hook_receiver.event", hook_event=payload.get("event"))
        await on_event(payload)
    except asyncio.TimeoutError:
        log.warning("git_hook_receiver.timeout")
    except Exception as exc:
        log.warning("git_hook_receiver.error", error=str(exc))
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def serve_socket(sock_path: Path, on_event: OnEventFn) -> None:
    """Listen on *sock_path* for git hook events indefinitely.

    Removes any stale socket file first so bind always succeeds on restart.
    Designed to be launched with ``asyncio.ensure_future()`` from mcp_bridge.
    """
    sock_path.unlink(missing_ok=True)

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_connection(r, w, on_event),
        path=str(sock_path),
    )
    log.info("git_hook_receiver.listening", sock=str(sock_path))
    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------
# Hook installer (called by `libucks install-hooks`)
# ---------------------------------------------------------------------------

_HOOK_EVENTS = ["post-commit", "post-checkout", "post-rewrite"]
# Absolute path to the libucks binary is baked in so git's sterile hook shell
# doesn't need the venv on PATH.  LIBUCKS_REPO_PATH ensures the correct socket
# is found when the indexed subdirectory differs from the git root.
_HOOK_LINE = "LIBUCKS_REPO_PATH={libucks_path} {libucks_bin} hook {event} \"$@\" || true"


def find_git_root(path: Path) -> Path:
    """Walk up from *path* until a directory containing ``.git/`` is found.

    Raises RuntimeError if no git root exists above *path*.
    """
    current = path.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"No .git directory found above {path}")
        current = parent


_LIBUCKS_MARKER = "libucks hook"


def install_hooks(
    libucks_path: Path,
    git_root: Path | None = None,
    force: bool = False,
    libucks_bin: str | None = None,
) -> tuple[list[str], Path]:
    """Append libucks trigger lines to .git/hooks/.

    *libucks_path* is the directory libucks tracks (contains ``.libucks/``).
    *git_root*     is where ``.git/`` lives; auto-detected by walking up from
                   *libucks_path* when not supplied.
    *force*        remove any existing libucks trigger lines first, then
                   re-install fresh ones (use to fix stale/mismatched hooks).
    *libucks_bin*  absolute path to the libucks executable; resolved via
                   ``shutil.which`` when omitted.

    Rules (normal mode):
    - If the hook file does not exist: create it with a ``#!/bin/sh`` shebang.
    - If it exists but already contains our trigger: skip (idempotent).
    - Always appends — never overwrites non-libucks content.
    - Sets executable bit on every modified file.

    Returns ``(modified_hook_names, hooks_dir)``.
    """
    import shutil as _shutil
    if libucks_bin is None:
        libucks_bin = _shutil.which("libucks") or "libucks"

    if git_root is None:
        git_root = find_git_root(libucks_path)

    hooks_dir = git_root / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    abs_libucks = libucks_path.resolve()
    modified: list[str] = []
    for event in _HOOK_EVENTS:
        trigger = _HOOK_LINE.format(
            libucks_path=abs_libucks, libucks_bin=libucks_bin, event=event
        )
        hook_file = hooks_dir / event

        if hook_file.exists():
            existing = hook_file.read_text()

            if force:
                # Strip ALL existing libucks trigger lines (any format/path).
                clean_lines = [
                    ln for ln in existing.splitlines()
                    if _LIBUCKS_MARKER not in ln
                ]
                existing = "\n".join(clean_lines).rstrip("\n")

            elif trigger in existing:
                log.debug("git_hook_receiver.hook_already_installed", hook_event=event)
                continue

            hook_file.write_text(existing.rstrip("\n") + "\n" + trigger + "\n")
        else:
            hook_file.write_text(f"#!/bin/sh\n{trigger}\n")

        # Always ensure executable — covers both new files and appended existing ones.
        hook_file.chmod(0o755)
        modified.append(event)
        log.info("git_hook_receiver.hook_installed", hook_event=event, path=str(hook_file))

    return modified, hooks_dir
