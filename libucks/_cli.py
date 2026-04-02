"""CLI entry point — lives inside the package so the console script works from any directory."""
import asyncio
import json
import socket
import subprocess
from pathlib import Path

import click


def _find_repo_root() -> Path:
    """Return the git repo root for cwd, or cwd itself if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return Path.cwd()


@click.group()
@click.version_option(version="0.1.0", prog_name="libucks")
def cli():
    """libucks — Librarian Buckets, local AI memory server for coding agents."""


@cli.command("init")
@click.option("--local", "local_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help="Path to a local repository to index.")
def init_cmd(local_path: Path):
    """Seed libucks buckets from a local repository."""
    from libucks.init_orchestrator import InitOrchestrator

    orchestrator = InitOrchestrator(local_path)
    asyncio.run(orchestrator.run())


@cli.command("serve")
def serve_cmd():
    """Start the libucks MCP server over stdio."""
    from libucks.mcp_bridge import serve
    asyncio.run(serve())


@cli.command("install-hooks")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
def install_hooks_cmd(repo_path: Path | None):
    """Append libucks git hook triggers to .git/hooks/ (never overwrites)."""
    from libucks.git_hook_receiver import install_hooks

    target = repo_path or _find_repo_root()
    modified = install_hooks(target)
    if modified:
        click.echo(f"Installed hooks: {', '.join(modified)}")
    else:
        click.echo("All hooks already installed — nothing changed.")


@cli.command("hook")
@click.argument("event")
@click.argument("args", nargs=-1)
def hook_cmd(event: str, args: tuple):
    """Send a git hook event to the running libucks server (called by git hooks)."""
    repo_path = _find_repo_root()
    sock_path = repo_path / ".libucks" / "server.sock"
    if not sock_path.exists():
        return  # server not running — silent exit so git is never blocked

    payload = json.dumps({"event": event, "args": list(args)}).encode()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            s.connect(str(sock_path))
            s.sendall(payload)
    except Exception:
        pass  # never block git
