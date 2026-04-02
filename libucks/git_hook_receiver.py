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
_HOOK_LINE = "libucks hook {event} \"$@\" || true"


def install_hooks(repo_path: Path) -> list[str]:
    """Append libucks trigger lines to .git/hooks/.

    Rules:
    - If the hook file does not exist: create it with a ``#!/bin/sh`` shebang.
    - If it exists but already contains our trigger: skip (idempotent).
    - Always appends — never overwrites existing content.
    - Sets executable bit on newly created files.

    Returns the list of hook names that were modified.
    """
    hooks_dir = repo_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    modified: list[str] = []
    for event in _HOOK_EVENTS:
        trigger = _HOOK_LINE.format(event=event)
        hook_file = hooks_dir / event

        if hook_file.exists():
            existing = hook_file.read_text()
            if trigger in existing:
                log.debug("git_hook_receiver.hook_already_installed", hook_event=event)
                continue
            hook_file.write_text(existing.rstrip("\n") + "\n" + trigger + "\n")
        else:
            hook_file.write_text(f"#!/bin/sh\n{trigger}\n")
            hook_file.chmod(0o755)

        modified.append(event)
        log.info("git_hook_receiver.hook_installed", hook_event=event, path=str(hook_file))

    return modified
