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
    from libucks.config import Config
    from libucks.init_orchestrator import InitOrchestrator
    from libucks.thinking import create_strategy

    cfg = Config.load(local_path)
    strategy = create_strategy(cfg)
    orchestrator = InitOrchestrator(local_path, strategy=strategy)
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


@cli.command("train-adapter")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--creative", is_flag=True, default=False,
              help="Use contrastive training with multi-perspective triplets and hard negatives.")
@click.option("--epochs", default=1, show_default=True, help="Number of training epochs.")
def train_adapter_cmd(repo_path: Path | None, creative: bool, epochs: int):
    """Train the CommunicationAdapter to align Librarian latents with teacher targets.

    With --creative: generates multi-perspective triplets (Summary, Logic Flow,
    Dependency Map) and mines hard negatives for InfoNCE contrastive training.

    Without --creative: uses basic MSE alignment between adapter output and
    teacher target latents.

    Saves trained weights to <repo>/.libucks/adapter.pt.
    """
    target = repo_path or _find_repo_root()
    asyncio.run(_run_train_adapter(target, creative=creative, epochs=epochs))


async def _run_train_adapter(repo_path: Path, creative: bool, epochs: int) -> None:
    from libucks.config import Config
    from libucks.thinking import create_strategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.storage.bucket_store import BucketStore
    from libucks.storage.bucket_registry import BucketRegistry

    cfg = Config.load(repo_path)

    if cfg.model.strategy != "latent":
        raise click.ClickException(
            "train-adapter requires strategy='latent' in .libucks/config.toml.\n"
            "The CommunicationAdapter operates on torch.Tensor hidden states; "
            "TextStrategy returns strings and cannot be used for training.\n\n"
            "Set the following in your config and re-run:\n\n"
            "  [model]\n"
            "  strategy = \"latent\"\n"
            "  local_model = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
            "  device = \"mps\"  # or cuda / cpu"
        )

    registry_path = repo_path / cfg.paths.registry_file
    bucket_dir = repo_path / ".libucks"

    registry = BucketRegistry(registry_path)
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)

    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        click.echo("No buckets found — run `libucks init` first.", err=True)
        return

    from transformers import AutoConfig as _AutoConfig
    _hidden_dim = _AutoConfig.from_pretrained(cfg.model.local_model).hidden_size
    adapter = CommunicationAdapter(hidden_dim=_hidden_dim)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")

    from libucks.thinking.model_manager import ModelManager as _MM
    _training_device = _MM._resolve_device(cfg.model.device)
    adapter = adapter.to(_training_device)

    if creative:
        click.echo(f"[libucks] Creative contrastive training on {len(bucket_ids)} buckets "
                   f"for {epochs} epoch(s)...")
        await _train_creative(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir)
    else:
        click.echo(f"[libucks] Basic MSE training on {len(bucket_ids)} buckets "
                   f"for {epochs} epoch(s)...")
        await _train_basic(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir)


async def _train_creative(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir):
    """Creative mode: multi-perspective + hard negatives + InfoNCE."""
    import os
    from libucks.thinking import create_strategy
    from libucks.thinking.latent_strategy import LatentStrategy
    from libucks.thinking.text_strategy import TextStrategy
    from libucks.thinking.training.data_generator import MultiPerspectiveDataGenerator
    from libucks.thinking.training.train_adapter import ContrastiveAdapterTrainer

    text_strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    # Use latent strategy only if configured; otherwise encode via text (fallback)
    if cfg.model.strategy == "latent":
        latent_strategy = create_strategy(cfg)
    else:
        click.echo("[libucks] Warning: strategy='text' — encoding via TextStrategy passthrough.",
                   err=True)
        latent_strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    generator = MultiPerspectiveDataGenerator(
        text_strategy=text_strategy,
        latent_strategy=latent_strategy,
        registry=registry,
        store=store,
    )
    trainer = ContrastiveAdapterTrainer(adapter, temperature=0.07, lr=1e-4)

    samples = []
    for i, bucket_id in enumerate(bucket_ids, 1):
        click.echo(f"  Generating sample {i}/{len(bucket_ids)}: {bucket_id}")
        try:
            sample = await generator.generate(bucket_id)
            samples.append(sample)
        except Exception as exc:
            click.echo(f"  Skipped {bucket_id}: {exc}", err=True)

    if not samples:
        click.echo("No training samples generated.", err=True)
        return

    losses = trainer.train(samples, num_epochs=epochs)
    trainer.save(bucket_dir / "adapter.pt")

    first = sum(losses[:5]) / min(5, len(losses))
    last = sum(losses[-5:]) / min(5, len(losses))
    click.echo(f"Training complete. Loss: {first:.4f} → {last:.4f}. "
               f"Saved to {bucket_dir / 'adapter.pt'}")


async def _train_basic(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir):
    """Basic mode: MSE between adapter mean-pooled output and target latent."""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from libucks.thinking import create_strategy
    from libucks.thinking.text_strategy import TextStrategy
    from libucks.thinking.training.data_generator import (
        MultiPerspectiveDataGenerator, PERSPECTIVE_PROMPTS
    )

    text_strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    if cfg.model.strategy == "latent":
        latent_strategy = create_strategy(cfg)
    else:
        latent_strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    generator = MultiPerspectiveDataGenerator(
        text_strategy=text_strategy,
        latent_strategy=latent_strategy,
        registry=registry,
        store=store,
    )
    optimizer = AdamW(adapter.parameters(), lr=1e-4)
    _device = next(adapter.parameters()).device

    for epoch in range(epochs):
        total_loss = 0.0
        for i, bucket_id in enumerate(bucket_ids, 1):
            try:
                sample = await generator.generate(bucket_id)
            except Exception as exc:
                click.echo(f"  Skipped {bucket_id}: {exc}", err=True)
                continue

            latents = [t.clone().detach().to(_device, torch.float32) for t in sample.librarian_latents]
            latents = [t.to(torch.float32) for t in latents]
            optimizer.zero_grad()
            output = adapter(latents)
            anchor = F.normalize(output.mean(dim=0), dim=0)
            target = F.normalize(sample.target_latent.clone().detach().to(_device, torch.float32).mean(dim=0), dim=0)
            loss = 1.0 - torch.dot(anchor, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            click.echo(f"  Epoch {epoch+1} [{i}/{len(bucket_ids)}] loss={loss.item():.4f}")

    torch.save(adapter.state_dict(), bucket_dir / "adapter.pt")
    click.echo(f"Basic training complete. Saved to {bucket_dir / 'adapter.pt'}")


@cli.command("query")
@click.argument("query_text")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--top-k", default=3, show_default=True, help="Number of buckets to consult.")
def query_cmd(query_text: str, repo_path: Path | None, top_k: int):
    """Run a single query against the local memory engine and print the answer.

    Bypasses the MCP server entirely — no 60-second timeout. Useful for
    validating the full inference pipeline from the terminal.

    Example:
        libucks query "How does the authentication module work?"
    """
    target = repo_path or _find_repo_root()
    asyncio.run(_run_query(target, query_text, top_k))


async def _run_query(repo_path: Path, query_text: str, top_k: int) -> None:
    import sys
    from libucks.config import Config
    from libucks.thinking import create_strategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.central_agent import CentralAgent
    from libucks.librarian import Librarian
    from libucks.query_orchestrator import QueryOrchestrator
    from libucks.translator import Translator

    cfg = Config.load(repo_path)
    registry_path = repo_path / cfg.paths.registry_file
    bucket_dir = repo_path / ".libucks"
    bucket_store_dir = repo_path / cfg.paths.bucket_dir

    click.echo(f"[libucks] repo={repo_path}  strategy={cfg.model.strategy}", err=True)

    registry = BucketRegistry(registry_path)
    registry.load()
    store = BucketStore(bucket_store_dir)

    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        click.echo("No buckets found — run `libucks init` first.", err=True)
        return

    click.echo(f"[libucks] {len(bucket_ids)} buckets loaded", err=True)

    # Load embedding model (suppress stdout during model loading)
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    finally:
        sys.stdout = _real_stdout

    click.echo("[libucks] embedding model ready, loading strategy...", err=True)
    strategy = create_strategy(cfg)
    click.echo("[libucks] strategy ready", err=True)

    agent = CentralAgent(registry, cfg, embed_fn=embedder.embed)
    librarians: dict[str, Librarian] = {}
    for bucket_id in bucket_ids:
        lib = Librarian(
            bucket_id=bucket_id,
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
        )
        librarians[bucket_id] = lib
        agent.register_librarian(bucket_id, lib)

    adapter = None
    if cfg.model.strategy == "latent":
        import torch
        resolved_device = cfg.model.device if cfg.model.device != "auto" else "mps"
        adapter = CommunicationAdapter(hidden_dim=strategy.hidden_dim)
        adapter.load_saved_weights(bucket_dir / "adapter.pt")
        # dtype must match the model's output dtype. ModelManager loads Qwen in
        # float16 on MPS; float32 adapter parameters cause an MPS broadcast error.
        adapter_dtype = torch.float16 if resolved_device == "mps" else None
        adapter = adapter.to(device=resolved_device, dtype=adapter_dtype)

    translator = Translator(strategy, adapter=adapter)

    orchestrator = QueryOrchestrator(
        central_agent=agent,
        librarians=librarians,
        embed_fn=embedder.embed,
        top_k=top_k,
    )

    click.echo(f"[libucks] routing: \"{query_text}\"", err=True)
    representations = await orchestrator.query(query_text)
    click.echo(f"[libucks] {len(representations)} representations, synthesizing...", err=True)

    answer = await translator.synthesize(query_text, representations)

    # Answer goes to stdout so it can be piped / captured cleanly.
    click.echo(answer)


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
