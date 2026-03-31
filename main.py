import asyncio
from pathlib import Path

import click


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
    import asyncio
    from libucks.mcp_bridge import serve
    asyncio.run(serve())


if __name__ == "__main__":
    cli()
