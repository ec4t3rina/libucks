"""CLI entry point — lives inside the package so the console script works from any directory."""
import asyncio
import json
import socket
import subprocess
from pathlib import Path

import click

# Load .env (cwd or any parent) before anything reads env vars like ANTHROPIC_API_KEY.
# Optional dep — falls back to relying on shell-exported env vars if missing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _load_lora_weights(strategy, bucket_dir: Path, device: str) -> None:
    """Inject LoRA structure into the Base receiver and load saved delta weights.

    Also loads W_a and soft_mean from w_a.pt (saved during train-adapter) and
    attaches them to strategy so decode() can mirror the training transform:
      soft_prompt → subtract soft_mean → @ W_a → norm-rescale → frame → generate

    Safe to call even if lora_receiver.pt does not exist — it silently skips.
    Must be called AFTER strategy._mgr.load_base_model().
    """
    lora_path = bucket_dir / "lora_receiver.pt"
    if not lora_path.exists():
        click.echo(f"[libucks] lora_receiver.pt not found at {lora_path.resolve()}", err=True)
        return
    import datetime, os, torch
    from libucks.thinking.training.lora_trainer import _inject_lora, _LORA_TARGETS
    base_model = strategy._mgr.get_base_model()
    _inject_lora(base_model, _LORA_TARGETS, r=16, alpha=16.0)
    state = torch.load(lora_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state, strict=False)
    click.echo(
        f"[libucks] LoRA loaded: {lora_path.resolve()} "
        f"(mtime={datetime.datetime.fromtimestamp(os.path.getmtime(lora_path))})",
        err=True,
    )

    w_a_path = bucket_dir / "w_a.pt"
    if w_a_path.exists():
        wa_state = torch.load(w_a_path, map_location=device, weights_only=True)
        strategy._W_a = wa_state["W_a"].to(device=device, dtype=torch.float32)
        strategy._soft_mean = wa_state["soft_mean"].to(device=device, dtype=torch.float32)
        click.echo(
            f"[libucks] W_a loaded: {w_a_path.resolve()} "
            f"(mtime={datetime.datetime.fromtimestamp(os.path.getmtime(w_a_path))}, "
            f"shape={tuple(strategy._W_a.shape)})",
            err=True,
        )
    else:
        click.echo(f"[libucks] w_a.pt not found at {w_a_path.resolve()} — run train-adapter", err=True)


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
@click.option("--train", "do_train", is_flag=True, default=False,
              help="After indexing, train the CommunicationAdapter and LoRA receiver on the new buckets.")
@click.option("--no-teacher", "no_teacher", is_flag=True, default=False,
              help="Self-supervised training — skip Anthropic teacher API calls. "
                   "Use when ANTHROPIC_API_KEY is unavailable.")
@click.option("--epochs", default=5, show_default=True,
              help="Training epochs (only used when --train is passed).")
def init_cmd(local_path: Path, do_train: bool, no_teacher: bool, epochs: int):
    """Seed libucks buckets from a local repository."""
    from libucks.config import Config
    from libucks.init_orchestrator import InitOrchestrator
    from libucks.thinking import create_strategy

    cfg = Config.load(local_path)
    strategy = create_strategy(cfg)
    orchestrator = InitOrchestrator(local_path, strategy=strategy)
    asyncio.run(orchestrator.run())

    # Register this repo in the known-repos list for discoverability.
    libucks_home = Path.home() / ".libucks"
    libucks_home.mkdir(exist_ok=True)
    known = libucks_home / "known_repos.txt"
    resolved = str(local_path.resolve())
    existing = known.read_text().splitlines() if known.exists() else []
    if resolved not in existing:
        with known.open("a") as f:
            f.write(resolved + "\n")

    if do_train:
        click.confirm(
            f"Training the LoRA receiver can take significant time depending on your hardware. "
            f"Proceed with {epochs} epoch(s)?",
            abort=True,
        )
        click.echo(f"[libucks] --train: starting adapter + LoRA receiver training ({epochs} epoch(s))...")
        asyncio.run(_run_train_adapter(
            local_path,
            creative=False,
            no_teacher=no_teacher,
            train_receiver=True,
            receiver_only=False,
            epochs=epochs,
        ))


@cli.command("use")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
def use_cmd(repo_path: Path):
    """Set the active repository for the libucks MCP server.

    After running this command, reconnect the MCP server in Claude Code
    (Settings → MCP → Reconnect) to apply the change.
    """
    libucks_home = Path.home() / ".libucks"
    libucks_home.mkdir(exist_ok=True)
    active_repo_file = libucks_home / "active_repo"
    resolved = repo_path.resolve()
    active_repo_file.write_text(str(resolved))
    click.echo(f"Active repo → {resolved}")
    click.echo("Reconnect the MCP server in Claude Code to apply.")


@cli.command("serve")
def serve_cmd():
    """Start the libucks MCP server over stdio."""
    from libucks.mcp_bridge import serve
    asyncio.run(serve())


@cli.command("install-hooks")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path libucks tracks (contains .libucks/). Defaults to cwd's git repo root.")
@click.option("--git-root", "git_root_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Override git root (where .git/ lives). Auto-detected when omitted.")
@click.option("--force", is_flag=True, default=False,
              help="Remove any existing libucks hook lines and re-install fresh ones.")
def install_hooks_cmd(repo_path: Path | None, git_root_path: Path | None, force: bool):
    """Append libucks git hook triggers to .git/hooks/ (never overwrites).

    The git root is auto-detected by walking up from --repo.  Use --git-root to
    override when a nested .git folder causes auto-detection to stop too early.
    Hooks bake in LIBUCKS_REPO_PATH so the hook always finds the correct socket
    even when --repo is a subdirectory of the git root.
    Use --force to fix stale or mismatched hooks from a previous install.
    """
    from libucks.git_hook_receiver import install_hooks, find_git_root

    libucks_path = (repo_path or _find_repo_root()).resolve()

    if git_root_path is not None:
        git_root = git_root_path.resolve()
    else:
        try:
            git_root = find_git_root(libucks_path)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))

    import shutil as _shutil, sys as _sys
    libucks_bin = (
        _shutil.which("libucks")
        or str(Path(_sys.argv[0]).resolve())
    )

    hooks_dir = git_root / ".git" / "hooks"
    click.echo(f"Git root   : {git_root}")
    click.echo(f"Hooks dir  : {hooks_dir}")
    click.echo(f"Tracking   : {libucks_path}")
    click.echo(f"Binary     : {libucks_bin}")

    modified, _ = install_hooks(libucks_path, git_root=git_root, force=force, libucks_bin=libucks_bin)
    if modified:
        click.echo(f"Installed hooks: {', '.join(modified)}")
    else:
        click.echo("All hooks already installed — nothing changed.")


@cli.command("train-adapter")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--creative", is_flag=True, default=False,
              help="Use contrastive training with multi-perspective triplets and hard negatives.")
@click.option("--no-teacher", "no_teacher", is_flag=True, default=False,
              help="Self-supervised mode: skip Anthropic teacher calls and encode bucket "
                   "prose directly via the local model. Useful when no API credits are available.")
@click.option("--train-receiver", "train_receiver", is_flag=True, default=False,
              help="Phase 2: after adapter training, fine-tune the Base receiver with "
                   "LoRA using L_task + L_sep (Interlat-Lite). Saves lora_receiver.pt.")
@click.option("--receiver-only", "receiver_only", is_flag=True, default=False,
              help="Skip Phase 1 entirely. Load adapter.pt from disk and run only Phase 2 "
                   "LoRA receiver training. Requires a previously saved adapter.pt.")
@click.option("--epochs", default=1, show_default=True, help="Number of training epochs.")
@click.option("--accum-steps", "accum_steps", default=1, show_default=True,
              help="Gradient accumulation steps for LoRA receiver training. "
                   "Default=1: every sample triggers an optimizer step (max steps per epoch). "
                   "Increase only on large repos (>500 buckets) to reduce gradient variance.")
@click.option("--lr", "lora_lr", default=2e-4, show_default=True,
              help="AdamW learning rate for LoRA receiver training. "
                   "2e-4 is standard for LoRA; 5e-5 is too small for <1000 optimizer steps.")
@click.option("--query-dropout-rate", "query_dropout_rate", default=0.5, show_default=True,
              help="Fraction of training steps where the query token prefix is dropped (Q=0). "
                   "L_sep only fires on Q=0 steps, so this controls how much latent-grounding "
                   "signal the model gets. CLAUDE.md mandates 0.5 — values < 0.2 starve L_sep "
                   "and the latent collapses (the May-14 click weights were trained at 0.1 and "
                   "hallucinate). Range: 0.0–1.0.")
@click.option("--lora-r", "lora_r", default=16, show_default=True,
              help="LoRA rank. Must match the rank used by mcp_bridge to load weights "
                   "(currently 16). Larger r = more LoRA capacity, slower training.")
@click.option("--lora-alpha", "lora_alpha", default=16.0, show_default=True,
              help="LoRA scaling factor. Effective scale is alpha/r; equal alpha/r keeps "
                   "the initial perturbation small.")
@click.option("--qa-per-bucket", "qa_per_bucket", default=1, show_default=True,
              help="Number of QA pairs to synthesize per bucket via the teacher. More pairs = "
                   "more training data per epoch (qa_per_bucket=3 with 43 click buckets = 129 "
                   "examples per epoch instead of 43). Use 3 as a starting point for the "
                   "post-fix retraining; 5 if 3 doesn't converge.")
@click.option("--sep-lambda", "sep_lambda", default=0.1, show_default=True,
              help="Weight on L_sep (latent-grounding loss) in Phase 2. 0.1 is the historical "
                   "default that works on 0.5B-1.5B receivers; larger receivers have stronger "
                   "task priors that drown out the latent gradient at 0.1 -- bump to 0.3 or "
                   "higher for 3B+. Range: 0.0-10.0.")
@click.option("--hybrid-train", "hybrid_train", is_flag=True, default=False,
              help="Phase B: train LoRA with verbatim source prepended to the input "
                   "frame during 50% of steps (matches inference-time hybrid retrieval). "
                   "Fixes the Q2-style decoder collapse seen when verbatim is added at "
                   "inference but LoRA was trained without it. Requires --train-receiver "
                   "or --receiver-only.")
def train_adapter_cmd(
    repo_path: Path | None,
    creative: bool,
    no_teacher: bool,
    train_receiver: bool,
    receiver_only: bool,
    epochs: int,
    accum_steps: int,
    lora_lr: float,
    query_dropout_rate: float,
    lora_r: int,
    lora_alpha: float,
    qa_per_bucket: int,
    sep_lambda: float,
    hybrid_train: bool,
):
    """Train the CommunicationAdapter to align Librarian latents with teacher targets.

    With --creative: generates multi-perspective triplets (Summary, Logic Flow,
    Dependency Map) and mines hard negatives for InfoNCE contrastive training.

    Without --creative: uses basic MSE alignment between adapter output and
    teacher target latents.

    With --no-teacher: self-supervised — encodes bucket prose directly via the
    local model without calling the Anthropic teacher API.

    With --train-receiver: Phase 2 — fine-tunes the Base receiver with LoRA using
    L_task - lambda*L_sep (Interlat-Lite). Requires --no-teacher or valid API key.
    Saves lora_receiver.pt alongside adapter.pt.

    Saves trained weights to <repo>/.libucks/adapter.pt.
    """
    target = repo_path or _find_repo_root()
    if not (0.0 <= query_dropout_rate <= 1.0):
        raise click.ClickException(
            f"--query-dropout-rate must be in [0.0, 1.0]; got {query_dropout_rate}"
        )
    if qa_per_bucket < 1 or qa_per_bucket > 10:
        raise click.ClickException(
            f"--qa-per-bucket must be in [1, 10]; got {qa_per_bucket}"
        )
    if not (0.0 <= sep_lambda <= 10.0):
        raise click.ClickException(
            f"--sep-lambda must be in [0.0, 10.0]; got {sep_lambda}"
        )
    asyncio.run(_run_train_adapter(
        target, creative=creative, no_teacher=no_teacher,
        train_receiver=train_receiver, receiver_only=receiver_only, epochs=epochs,
        accum_steps=accum_steps, lora_lr=lora_lr,
        query_dropout_rate=query_dropout_rate,
        lora_r=lora_r, lora_alpha=lora_alpha,
        qa_per_bucket=qa_per_bucket,
        sep_lambda=sep_lambda,
        hybrid_train=hybrid_train,
    ))


async def _run_train_adapter(
    repo_path: Path, creative: bool, no_teacher: bool, train_receiver: bool,
    receiver_only: bool = False, epochs: int = 1, accum_steps: int = 8, lora_lr: float = 1e-4,
    query_dropout_rate: float = 0.5, lora_r: int = 16, lora_alpha: float = 16.0,
    qa_per_bucket: int = 1, sep_lambda: float = 0.1, hybrid_train: bool = False,
) -> None:
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
    _base_dim = _AutoConfig.from_pretrained(cfg.model.base_model).hidden_size
    adapter = CommunicationAdapter(hidden_dim=_hidden_dim, output_dim=_base_dim,
                                   output_len=cfg.model.output_len)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")

    from libucks.thinking.model_manager import ModelManager as _MM
    _training_device = _MM._resolve_device(cfg.model.device)
    adapter = adapter.to(_training_device)

    if not receiver_only:
        if no_teacher:
            click.echo(f"[libucks] Self-supervised training (no teacher) on {len(bucket_ids)} buckets "
                       f"for {epochs} epoch(s)...")
            await _train_no_teacher(cfg, store, bucket_ids, adapter, epochs, bucket_dir)
        elif creative:
            click.echo(f"[libucks] Creative contrastive training on {len(bucket_ids)} buckets "
                       f"for {epochs} epoch(s)...")
            await _train_creative(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir)
        else:
            click.echo(f"[libucks] Basic MSE training on {len(bucket_ids)} buckets "
                       f"for {epochs} epoch(s)...")
            await _train_basic(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir)
    else:
        click.echo("[libucks] --receiver-only: skipping Phase 1, using adapter.pt from disk", err=True)

    if train_receiver or receiver_only:
        click.echo(f"\n[libucks] Phase 2: LoRA receiver training on {len(bucket_ids)} buckets "
                   f"for {epochs} epoch(s)...")
        await _train_lora_receiver(cfg, store, bucket_ids, adapter, epochs, bucket_dir,
                                   no_teacher=no_teacher, registry=registry,
                                   accum_steps=accum_steps, lora_lr=lora_lr,
                                   query_dropout_rate=query_dropout_rate,
                                   lora_r=lora_r, lora_alpha=lora_alpha,
                                   qa_per_bucket=qa_per_bucket,
                                   sep_lambda=sep_lambda,
                                   hybrid_train=hybrid_train)


async def _train_no_teacher(cfg, store, bucket_ids, adapter, epochs, bucket_dir):
    """Self-supervised mode: encode bucket prose directly — no Anthropic teacher API needed.

    Loss: L_align (cosine toward own hidden-state mean) + L_spread (margin repulsion
    against recent buckets' outputs via a rolling queue).  L_spread is the critical
    addition — without it the adapter converges to a single mean embedding for all
    buckets, making the LoRA receiver unable to distinguish any latent from any other.
    """
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from libucks.thinking import create_strategy
    from libucks.thinking.training.data_generator import PERSPECTIVE_PROMPTS

    latent_strategy = create_strategy(cfg)
    optimizer = AdamW(adapter.parameters(), lr=1e-4)
    _device = next(adapter.parameters()).device
    _SPREAD_MARGIN = 0.5   # target minimum L2 distance between any two bucket outputs
    _SPREAD_LAMBDA = 1.0   # weight on L_spread relative to L_align
    _NEG_QUEUE_SIZE = 16   # rolling buffer of recent outputs used as negatives

    all_losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        neg_queue: list[torch.Tensor] = []   # detached normalized outputs, newest last

        for i, bucket_id in enumerate(bucket_ids, 1):
            try:
                _, prose = store.read(bucket_id)

                # Encode prose from three perspectives (no API call needed)
                latents: list[torch.Tensor] = []
                for prompt in PERSPECTIVE_PROMPTS:
                    hidden = await latent_strategy.reason(prompt, prose)
                    latents.append(hidden.clone().detach().to(_device, torch.float32))

                # Self-supervised target: first perspective projected to adapter output space.
                with torch.no_grad():
                    t_raw = latents[0].mean(dim=0)   # (instruct_dim,)
                    if adapter.output_proj is not None:
                        t_raw = adapter.output_proj(t_raw)  # (base_dim,)
                target = F.normalize(t_raw, dim=0)

                optimizer.zero_grad()
                output = adapter(latents)            # (K, base_dim)
                anchor = F.normalize(output.mean(dim=0), dim=0)
                loss = 1.0 - torch.dot(anchor, target)  # L_align

                # L_spread: margin repulsion against recent buckets in queue.
                # Forces the adapter to produce distinct outputs per bucket.
                if neg_queue:
                    negs = torch.stack(neg_queue)   # (N, base_dim)
                    # Pairwise L2 distances between anchor and each negative
                    dists = (anchor.unsqueeze(0) - negs).norm(dim=1)  # (N,)
                    spread_loss = F.relu(_SPREAD_MARGIN - dists).mean()
                    loss = loss + _SPREAD_LAMBDA * spread_loss

                loss.backward()
                optimizer.step()

                # Push normalized output to queue (detached — no grad).
                neg_queue.append(anchor.detach())
                if len(neg_queue) > _NEG_QUEUE_SIZE:
                    neg_queue.pop(0)

                val = loss.item()
                epoch_loss += val
                all_losses.append(val)
                click.echo(f"  Epoch {epoch+1} [{i}/{len(bucket_ids)}] bucket={bucket_id} loss={val:.4f}")
            except Exception as exc:
                click.echo(f"  Skipped {bucket_id}: {exc}", err=True)

    torch.save(adapter.state_dict(), bucket_dir / "adapter.pt")

    if all_losses:
        first5 = all_losses[:min(5, len(all_losses))]
        last5 = all_losses[-min(5, len(all_losses)):]
        click.echo(
            f"Self-supervised training complete. "
            f"Loss: {sum(first5)/len(first5):.4f} → {sum(last5)/len(last5):.4f}. "
            f"Saved to {bucket_dir / 'adapter.pt'}"
        )
    else:
        click.echo("No samples trained.", err=True)


async def _train_lora_receiver(cfg, store, bucket_ids, adapter, epochs, bucket_dir,
                               no_teacher: bool = False, registry=None, accum_steps: int = 8,
                               lora_lr: float = 1e-4, query_dropout_rate: float = 0.5,
                               lora_r: int = 16, lora_alpha: float = 16.0,
                               qa_per_bucket: int = 1, sep_lambda: float = 0.1,
                               hybrid_train: bool = False):
    """Phase 2 (Interlat-Lite): fine-tune the Base receiver with LoRA.

    Loss: L_total = L_task - λ_sep * L_sep + λ_align * L_align  (λ_sep=0.1, λ_align=0.05)

    For each bucket:
      1. Encode prose via reason() → soft-prompt via adapter.
      2. Sample curriculum rate r ~ U[0,1]; mix soft-prompt with plan token embeds.
      3. Frame with <bop>/<eop> boundary embeddings.
      4. Build wrong-path embeddings from a different bucket (for L_sep).
      5. Run LoRAReceiverTrainer.train_step().

    When no_teacher=False (default), calls the Anthropic teacher once per bucket
    during pre-compute to generate a natural-language description of the bucket's
    source code.  The LoRA receiver is then trained to decode latents into this
    English text rather than raw source code — producing conversational output.

    When no_teacher=True, falls back to _collect_source_text (code reconstruction).

    Saves LoRA delta weights (only lora_A / lora_B keys) to lora_receiver.pt.
    """
    import random
    import torch
    from libucks.thinking import create_strategy
    from libucks.thinking.curriculum import CurriculumMixer
    from libucks.thinking.training.data_generator import PERSPECTIVE_PROMPTS
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer, _inject_lora, _LORA_TARGETS

    latent_strategy = create_strategy(cfg)

    # Load the Base receiver model (separate from the Instruct encoder)
    click.echo("[libucks] loading Base receiver model...", err=True)
    latent_strategy._mgr.load_base_model(
        model_id=cfg.model.base_model,
        quantization=cfg.model.quantization,
        bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
        device=cfg.model.device,
        base_model_dtype=cfg.model.base_model_dtype,
    )
    click.echo("[libucks] Base receiver model ready", err=True)

    base_model = latent_strategy._mgr.get_base_model()
    base_tok = latent_strategy._mgr.get_base_tokenizer()
    embedding = base_model.model.embed_tokens
    device = next(base_model.parameters()).device
    model_dtype = embedding.weight.dtype   # float16 on MPS, float32 on CPU
    # Adapter was saved/loaded as float32. Cast to model_dtype so its internal
    # MHA biases don't collide with float16 latents on MPS. LoRA params stay
    # float32 (handled by LoRALinear.forward's .to(x.dtype)) — don't touch those.
    adapter = adapter.to(device=device, dtype=model_dtype)
    K = adapter.output_len  # 32

    # Build <bop> / <eop> boundary embeddings and assistant turn-start.
    # The assistant cue must be present in training to match decode() at inference:
    # decode() appends "<|im_start|>assistant\n" before generate() so the Instruct
    # model knows to produce a structured answer rather than a continuation fragment.
    bop_id = base_tok.convert_tokens_to_ids("<|im_start|>")
    eop_id = base_tok.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        bop_embed = embedding(torch.tensor([bop_id], device=device)).squeeze(0).detach().to(model_dtype)
        eop_embed = embedding(torch.tensor([eop_id], device=device)).squeeze(0).detach().to(model_dtype)
        _asst_ids = base_tok(
            "<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False,
        )["input_ids"].squeeze(0).to(device)
        asst_embed = embedding(_asst_ids).detach().to(model_dtype)  # (A, D)

    # Pre-encode all buckets so we don't call the model twice per step.
    # The latent encoding always uses the raw source text as the context
    # for reason() — that's the semantic content being compressed.
    # The CE training TARGET is either a teacher-generated English description
    # (default) or the raw source text (--no-teacher fallback).
    from libucks.thinking.training.data_generator import _collect_source_text

    # Instantiate the Anthropic teacher client once (reads ANTHROPIC_API_KEY
    # from the environment, same as the rest of the codebase).
    # Hard-fail upfront when key is missing rather than silently falling back to
    # source-code targets — the SDK's "no api_key" error is not AuthenticationError
    # so it slipped through the per-call except handler and produced 159 silent
    # fallback rows in qa_cache.json before this guard was added.
    teacher_client = None
    if not no_teacher:
        import os as _os
        if not _os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException(
                "ANTHROPIC_API_KEY not set. Either:\n"
                "  • create .env in the project root with ANTHROPIC_API_KEY=sk-ant-...\n"
                "  • or export ANTHROPIC_API_KEY=sk-ant-... in your shell\n"
                "  • or pass --no-teacher to train on raw source text (lower quality)."
            )
        try:
            import anthropic as _anthropic
            teacher_client = _anthropic.AsyncAnthropic()
            click.echo("[libucks] Anthropic teacher client ready for target generation", err=True)
        except ImportError:
            click.echo(
                "[libucks] Warning: anthropic package not found — falling back to source-code targets",
                err=True,
            )

    # Q&A prompt: the teacher generates N question/answer pairs so the LoRA
    # receiver learns to answer varied questions about the same bucket.
    # The question becomes the query conditioning prefix; the answer is the CE target.
    def _build_qa_prompt(n_pairs: int) -> str:
        if n_pairs <= 1:
            count_word = "ONE question"
            pair_word = "pair"
        else:
            count_word = f"{n_pairs} distinct questions"
            pair_word = "pairs"
        return (
            f"Given this source code, write {count_word} whose answers require specific "
            "facts from the code (function/class names, parameter signatures, constant "
            "values, return types, control flow). Each question must be ANSWERABLE only "
            "by reading this specific code — not by generic knowledge of the topic. "
            f"Make the {pair_word} cover different aspects of the code (different "
            "functions, branches, or invariants) — do not paraphrase the same question.\n\n"
            "Then write a concise 2-3 sentence plain English answer for each that "
            "explicitly names the relevant identifiers from the code.\n\n"
            "BAD example (too generic — answer comes from priors, not the code):\n"
            "QUESTION 1: How does this module handle errors?\n"
            "ANSWER 1: It catches exceptions and returns an error response.\n\n"
            "GOOD example (answer requires the code):\n"
            "QUESTION 1: What does load_config() return when the YAML file is missing?\n"
            "ANSWER 1: It returns the default Config() instance built from DEFAULT_SETTINGS, "
            "and logs a warning via logger.warning() rather than raising.\n\n"
            f"Format EXACTLY (numbered 1..{n_pairs}):\n"
            "QUESTION 1: <question>\nANSWER 1: <answer>\n"
            + ("QUESTION 2: <question>\nANSWER 2: <answer>\n..." if n_pairs > 1 else "")
        )

    _TEACHER_QA_PROMPT = _build_qa_prompt(qa_per_bucket)

    def _parse_qa_pairs(text: str, n_expected: int, fallback_q: str, fallback_a: str):
        """Parse up to n_expected QUESTION i:/ANSWER i: pairs. Returns list of (q, a).

        Tolerant: missing numbers, single-pair format (QUESTION:/ANSWER:), or
        partial responses all fall back gracefully so a teacher call that
        returned only 2 of 3 pairs still yields 2 usable pairs.
        """
        import re
        # Look for "QUESTION [optional number]:" markers; capture answer up to
        # the next QUESTION marker or end-of-text.
        pattern = re.compile(
            r"QUESTION\s*(\d*)\s*:\s*(.+?)\s*ANSWER\s*\d*\s*:\s*(.+?)(?=\n\s*QUESTION\s*\d*\s*:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        pairs: list[tuple[str, str]] = []
        for m in pattern.finditer(text):
            q = m.group(2).strip()
            a = m.group(3).strip()
            if q and a:
                pairs.append((q, a))
            if len(pairs) >= n_expected:
                break
        if not pairs:
            # Total parse failure — fall back to single placeholder pair.
            return [(fallback_q, fallback_a)]
        return pairs

    # ── Phase 1: parallel teacher API calls (I/O-bound, up to 5 concurrent) ──
    # Cache schema (v2):  {bucket_id: {"pairs": [[q, target], ...], "source": str}}
    # Backward compat:   legacy entries `[q, target, source]` are wrapped as
    #                    a single-pair entry on load.
    _qa_cache_path = bucket_dir / "qa_cache.json"
    _qa_cache: dict[str, dict] = {}

    if _qa_cache_path.exists():
        try:
            raw_cache = json.loads(_qa_cache_path.read_text())
            for k, v in raw_cache.items():
                if isinstance(v, list) and len(v) == 3 and all(isinstance(x, str) for x in v):
                    # Legacy 3-tuple: (question, target, source)
                    _qa_cache[k] = {"pairs": [[v[0], v[1]]], "source": v[2]}
                elif isinstance(v, dict) and "pairs" in v:
                    _qa_cache[k] = {
                        "pairs": [list(p) for p in v["pairs"]],
                        "source": v.get("source", ""),
                    }
                else:
                    click.echo(f"[libucks] Skipping malformed cache entry {k!r}", err=True)
            click.echo(
                f"[libucks] Loaded {len(_qa_cache)} bucket Q&A entries from cache "
                f"({sum(len(e['pairs']) for e in _qa_cache.values())} total pairs)",
                err=True,
            )
        except Exception as _e:
            click.echo(f"[libucks] Cache load failed ({_e}), re-fetching", err=True)
            _qa_cache = {}

    # A bucket is "uncached" if it's missing OR has fewer pairs than requested.
    uncached = [
        bid for bid in bucket_ids
        if bid not in _qa_cache or len(_qa_cache[bid]["pairs"]) < qa_per_bucket
    ]
    click.echo(
        f"[libucks] Phase 1: {len(_qa_cache)} cached, {len(uncached)} to fetch "
        f"(target {qa_per_bucket} pair(s)/bucket)...",
        err=True,
    )

    if teacher_client is not None and uncached:
        _sem = asyncio.Semaphore(1)

        async def _fetch_qa(bucket_id: str) -> tuple[str, list[tuple[str, str]], str | None]:
            front_matter, prose = store.read(bucket_id)
            source_text = _collect_source_text(front_matter, max_chars=1024) or prose or front_matter.domain_label
            fallback_q = PERSPECTIVE_PROMPTS[0]
            fallback_a = source_text or ""
            pairs: list[tuple[str, str]] = [(fallback_q, fallback_a)]
            if source_text:
                try:
                    async with _sem:
                        # Token budget scales roughly with pair count; ~200 tokens/pair.
                        max_toks = max(256, 200 * qa_per_bucket)
                        resp = await teacher_client.messages.create(
                            model=cfg.model.anthropic_model,
                            max_tokens=max_toks,
                            messages=[{"role": "user", "content": f"{_TEACHER_QA_PROMPT}\n\n{source_text}"}],
                        )
                        await asyncio.sleep(1.2)  # stay under 50 RPM limit
                    raw = resp.content[0].text.strip()
                    pairs = _parse_qa_pairs(raw, qa_per_bucket, fallback_q, fallback_a)
                    click.echo(
                        f"  Q&A {bucket_id}: {len(pairs)} pair(s)  "
                        f"Q1={pairs[0][0][:50]} | A1={pairs[0][1][:50]}...",
                        err=True,
                    )
                except (_anthropic.AuthenticationError,
                        _anthropic.APIConnectionError,
                        _anthropic.RateLimitError) as fatal_exc:
                    raise click.ClickException(
                        f"Anthropic API fatal error during LoRA receiver training — "
                        f"check ANTHROPIC_API_KEY and credit balance: {fatal_exc}"
                    ) from fatal_exc
                except Exception as transient_exc:
                    click.echo(
                        f"  Teacher call failed for {bucket_id}: {transient_exc} — using source text",
                        err=True,
                    )
            return bucket_id, pairs, source_text

        qa_results = await asyncio.gather(*[_fetch_qa(bid) for bid in uncached],
                                          return_exceptions=True)
        for item in qa_results:
            if isinstance(item, click.ClickException):
                raise item
            if isinstance(item, Exception):
                click.echo(f"  Skipped bucket (API error): {item}", err=True)
                continue
            bid, pairs, src = item
            _qa_cache[bid] = {"pairs": [list(p) for p in pairs], "source": src or ""}

        # Persist cache so re-runs skip API calls entirely.
        try:
            _qa_cache_path.write_text(json.dumps(_qa_cache, indent=2))
            click.echo(f"[libucks] Q&A cache saved to {_qa_cache_path}", err=True)
        except Exception as _e:
            click.echo(f"[libucks] Warning: could not save Q&A cache: {_e}", err=True)

    elif uncached:
        # no_teacher path — fill uncached with source text fallbacks
        for bucket_id in uncached:
            front_matter, prose = store.read(bucket_id)
            src = _collect_source_text(front_matter, max_chars=1024) or prose or front_matter.domain_label
            _qa_cache[bucket_id] = {
                "pairs": [[PERSPECTIVE_PROMPTS[0], src]],
                "source": src or "",
            }

    click.echo(f"[libucks] Phase 1 done — {len(_qa_cache)} Q&A pairs collected", err=True)

    # ── Phase 2: sequential GPU encoding (serialized by _device_lock on MPS) ──
    # Encoding is per-bucket (latent only depends on source); training units
    # are per-(bucket, pair) so qa_per_bucket > 1 multiplies steps per epoch.
    click.echo("[libucks] Phase 2: encoding latents sequentially...", err=True)
    bucket_soft: dict[str, torch.Tensor] = {}
    bucket_pairs: dict[str, list[tuple[str, str]]] = {}  # bid → [(question, target), ...]

    for bucket_id in bucket_ids:
        if bucket_id not in _qa_cache:
            continue
        entry = _qa_cache[bucket_id]
        source_text = entry["source"]
        pairs_raw = entry["pairs"]
        pairs = [(p[0], p[1]) for p in pairs_raw]
        try:
            hidden = await latent_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
            with torch.no_grad():
                soft = adapter([hidden.clone().detach().to(device, model_dtype)])
            bucket_soft[bucket_id] = soft.detach()
            bucket_pairs[bucket_id] = pairs
            click.echo(
                f"  encoded {bucket_id} → soft-prompt {tuple(soft.shape)}  "
                f"({len(pairs)} QA pair(s))",
                err=True,
            )
            if str(device).startswith("mps"):
                torch.mps.empty_cache()
        except click.ClickException:
            raise
        except Exception as exc:
            click.echo(f"  Skipped encoding {bucket_id}: {exc}", err=True)

    if not bucket_soft:
        click.echo("No buckets encoded — aborting LoRA training.", err=True)
        return

    encoded_ids = list(bucket_soft.keys())

    # Pre-flight data quality check — drop degenerate pairs per bucket. A bucket
    # survives if at least one of its pairs has a non-degenerate target.
    def _is_usable(target: str) -> bool:
        return not target.startswith("# /") and " " in target.strip()

    for bid in list(bucket_pairs.keys()):
        bucket_pairs[bid] = [(q, t) for (q, t) in bucket_pairs[bid] if _is_usable(t)]
        if not bucket_pairs[bid]:
            del bucket_pairs[bid]

    encoded_ids = [bid for bid in encoded_ids if bid in bucket_pairs]
    if not encoded_ids:
        raise click.ClickException(
            "All buckets have degenerate targets. Cannot train. "
            "Delete .libucks/qa_cache.json and re-run without --no-teacher."
        )

    # Build flat list of training units: each (bucket_id, question, target_text)
    # is one optimizer step. _total_opt_steps reflects unit count, not bucket count.
    _training_units: list[tuple[str, str, str]] = [
        (bid, q, t) for bid in encoded_ids for (q, t) in bucket_pairs[bid]
    ]
    click.echo(
        f"[libucks] {len(encoded_ids)} usable buckets × pairs = "
        f"{len(_training_units)} training units per epoch",
        err=True,
    )

    # Initialise LoRA receiver trainer now that unit count is final.
    # Warmup: 5% of total steps (floor 5). 20% was too long — wasted the first epoch
    # at near-zero lr when we need every step to count on small datasets.
    _total_opt_steps = max(1, (epochs * len(_training_units)) // accum_steps)
    _warmup_steps = max(5, _total_opt_steps // 20)
    trainer = LoRAReceiverTrainer(
        base_model, lora_r=lora_r, lora_alpha=lora_alpha, lr=lora_lr,
        warmup_steps=_warmup_steps, total_steps=_total_opt_steps,
        sep_lambda=sep_lambda,
    )
    click.echo(
        f"[libucks] LoRA trainer ready  lora_r={lora_r}  lora_alpha={lora_alpha}  "
        f"query_dropout_rate={query_dropout_rate}  sep_lambda={sep_lambda}",
        err=True,
    )

    # Compute W_a alignment matrix once (LatentMAS §A.1, ridge regression).
    # Maps hidden-state-space vectors h → input-embedding-space: e = h @ W_a.
    # W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in
    # LoRA only touches q_proj/v_proj, so embed_tokens and lm_head are frozen
    # throughout training — W_a remains valid for all steps and inference.
    with torch.no_grad():
        _W_in  = base_model.model.embed_tokens.weight.float()   # (V, D)
        _W_out = base_model.lm_head.weight.float()               # (V, D)
        _lam   = 1e-3
        _WtW   = _W_out.T @ _W_out                              # (D, D)
        _WtW.diagonal().add_(_lam)
        W_a = torch.linalg.solve(_WtW, _W_out.T @ _W_in)       # (D, D)
        W_a = W_a.to(device=device, dtype=torch.float32)
    del _W_in, _W_out, _WtW
    click.echo(
        f"[libucks] W_a alignment matrix computed  "
        f"shape={tuple(W_a.shape)}  "
        f"||W_a||_F={W_a.norm().item():.3f}",
        err=True,
    )

    # ── Per-slot norm rescale only (no centering) ─────────────────────── #
    # The legacy code subtracted a per-slot population mean before normalizing,
    # which we measured to actively re-collapse the (now diverse, post-residual-
    # fix) adapter output. With diverse adapter slots, soft_mean[k] is similar
    # to adapter_output[k] for "average" buckets, and the residuals after
    # subtraction become tiny noise that's correlated across slots — slot cos
    # jumps from ~0.50 (raw adapter) → ~0.97 (post-centering). Removing
    # centering keeps the adapter's diversity intact.
    # We still rescale each slot to embed_norm so soft-prompt magnitudes match
    # what attention is calibrated for.
    with torch.no_grad():
        _all_softs = torch.stack([bucket_soft[bid].float() for bid in encoded_ids])
        _target_norm = embedding.weight.data.float().norm(dim=-1).median()
        _pre_norm = _all_softs.norm(dim=-1).mean().item()
        for bid in encoded_ids:
            _sp = bucket_soft[bid].float()
            _sp_n = _sp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            bucket_soft[bid] = (_sp / _sp_n * _target_norm).to(model_dtype)
        _post_norm = torch.stack([bucket_soft[bid].float() for bid in encoded_ids]).norm(dim=-1).mean().item()
    # Set _soft_mean to zeros for backward-compatible w_a.pt schema; older
    # decode() paths that still subtract it become no-ops.
    _soft_mean = torch.zeros_like(_all_softs[0])
    click.echo(
        f"[libucks] Soft-prompts rescaled (no centering)  "
        f"pre_norm={_pre_norm:.2f}  post_norm={_post_norm:.4f}  target_norm={_target_norm:.4f}",
        err=True,
    )
    # Persist alignment artefacts so inference can mirror the training transform.
    # decode() applies: center (subtract soft_mean) → W_a project → norm-rescale.
    torch.save({"W_a": W_a.cpu(), "soft_mean": _soft_mean.cpu()},
               bucket_dir / "w_a.pt")
    click.echo("[libucks] W_a + soft_mean saved to w_a.pt", err=True)

    # ── Soft-prompt diversity diagnostic ──────────────────────────────── #
    if len(encoded_ids) >= 2:
        import random as _rng
        _diag_pairs = min(20, len(encoded_ids) * (len(encoded_ids) - 1) // 2)
        _ids = list(encoded_ids)
        _l2_vals, _jsd_vals, _raw_l2_vals, _raw_norms = [], [], [], []
        base_model.eval()
        with torch.no_grad():
            for _ in range(_diag_pairs):
                _a, _b = _rng.sample(_ids, 2)
                _raw_a = bucket_soft[_a].to(device, dtype=torch.float32)
                _raw_b = bucket_soft[_b].to(device, dtype=torch.float32)
                _raw_l2_vals.append((_raw_a - _raw_b).norm(dim=-1).mean().item())
                _raw_norms.append(_raw_a.norm(dim=-1).mean().item())
                _spa = (_raw_a @ W_a).to(model_dtype)
                _spb = (_raw_b @ W_a).to(model_dtype)
                _l2 = (_spa - _spb).norm(dim=-1).mean().item()
                _l2_vals.append(_l2)
                _bop = base_model.model.embed_tokens(
                    torch.tensor([base_tok.convert_tokens_to_ids("<bop>") if "<bop>" in base_tok.get_vocab() else 1], device=device)
                ).squeeze(0).to(model_dtype)
                _eop = base_model.model.embed_tokens(
                    torch.tensor([base_tok.convert_tokens_to_ids("<eop>") if "<eop>" in base_tok.get_vocab() else 2], device=device)
                ).squeeze(0).to(model_dtype)
                _emb_a = torch.cat([_bop.unsqueeze(0), _spa, _eop.unsqueeze(0)])
                _emb_b = torch.cat([_bop.unsqueeze(0), _spb, _eop.unsqueeze(0)])
                _la = base_model(inputs_embeds=_emb_a.unsqueeze(0), use_cache=False).logits.squeeze(0)[-1].float()
                _lb = base_model(inputs_embeds=_emb_b.unsqueeze(0), use_cache=False).logits.squeeze(0)[-1].float()
                import torch.nn.functional as _F
                _pa = _F.softmax(_la, dim=-1); _pb = _F.softmax(_lb, dim=-1)
                _m = 0.5 * (_pa + _pb)
                _eps = 1e-8
                _jsd = 0.5 * (_pa * (_pa.clamp(_eps).log() - _m.clamp(_eps).log())).sum() + \
                       0.5 * (_pb * (_pb.clamp(_eps).log() - _m.clamp(_eps).log())).sum()
                _jsd_vals.append(_jsd.item())
        base_model.train()
        click.echo(
            f"[libucks] Soft-prompt diversity  "
            f"raw_L2={sum(_raw_l2_vals)/len(_raw_l2_vals):.4f}  "
            f"raw_norm={sum(_raw_norms)/len(_raw_norms):.4f}  "
            f"aligned_L2={sum(_l2_vals)/len(_l2_vals):.4f}  "
            f"mean_JSD(base)={sum(_jsd_vals)/len(_jsd_vals):.6f}  "
            f"(n_pairs={_diag_pairs})",
            err=True,
        )

    # ── Phase B: hybrid-train verbatim cache ────────────────────────────────
    # When --hybrid-train is set, precompute per-bucket verbatim source token
    # ids once. At each training step, with p=0.5 the verbatim is prepended to
    # BOTH the correct and wrong inputs_embeds frames so the LoRA learns to
    # share attention between the soft prompt and a real text prefix — fixing
    # the Q2-style decoder collapse seen at inference when hybrid retrieval
    # injects verbatim into a LoRA that has never seen it.
    # Per-bucket char budget 1000 ≈ 250 tokens, matching inference where
    # 3000 chars / top-k=3 buckets ≈ 1000 chars per bucket.
    bucket_verbatim_ids: dict[str, "torch.Tensor"] = {}
    if hybrid_train:
        from libucks.thinking.training.data_generator import _collect_source_text as _cst
        click.echo("[libucks] hybrid-train: pre-tokenising verbatim source for each bucket...", err=True)
        for _bid in bucket_ids:
            try:
                _fm, _ = store.read(_bid)
                _text = _cst(_fm, max_chars=1000)
            except Exception:
                _text = ""
            if _text:
                _enc = base_tok(_text, return_tensors="pt", truncation=True,
                                max_length=256, add_special_tokens=False)
                _ids = _enc["input_ids"].squeeze(0).long().to(device)
                if _ids.shape[0] > 0:
                    bucket_verbatim_ids[_bid] = _ids
        click.echo(
            f"[libucks] hybrid-train: cached verbatim for "
            f"{len(bucket_verbatim_ids)}/{len(bucket_ids)} buckets",
            err=True,
        )

    all_task: list[float] = []
    all_sep: list[float] = []
    all_task_q0: list[float] = []   # task loss on query-dropped steps (latent-only)
    all_task_q1: list[float] = []   # task loss on query-present steps

    # Best-checkpoint tracking: sep can dip in late epochs from margin
    # saturation + lr decay. We want the highest-sep weights, not the final
    # ones, since the final weights aren't always best for downstream decoding.
    _best_mean_sep: float = -float("inf")
    _best_lora_state: dict | None = None
    _best_epoch: int = -1

    for epoch in range(epochs):
        _epoch_task_q0: list[float] = []
        _epoch_sep_q0:  list[float] = []
        _epoch_task_q1: list[float] = []
        # Shuffle units each epoch so the same bucket's pairs don't cluster
        # adjacent in the optimizer trajectory.
        _epoch_units = list(_training_units)
        random.shuffle(_epoch_units)
        for i, (bucket_id, unit_question, target_text) in enumerate(_epoch_units, 1):
            try:
                soft_prompt = bucket_soft[bucket_id]          # (K, D)

                # Skip degenerate targets defensively: the pre-flight filter
                # already dropped these, but a fresh teacher run on a tiny
                # bucket can still slip a one-word "answer" through.
                if target_text.startswith("# /") or " " not in target_text.strip():
                    continue

                # Tokenise source text → target_ids, truncated to 64 tokens
                enc = base_tok(target_text, return_tensors="pt", truncation=True, max_length=64)
                target_ids = enc["input_ids"].squeeze(0).long().to(device)  # (T,)
                if target_ids.shape[0] == 0:
                    click.echo(f"  Skipped {bucket_id}: target_text tokenised to 0 tokens "
                               f"(empty target_text = {repr(target_text[:40])})", err=True)
                    continue

                # Plan token embeddings: K tokens (truncate or pad with the last token)
                plan_ids = target_ids[:K]
                if plan_ids.shape[0] < K:
                    pad_val = plan_ids[-1] if plan_ids.shape[0] > 0 else torch.tensor(0, dtype=torch.long)
                    pad = pad_val.expand(K - plan_ids.shape[0])
                    plan_ids = torch.cat([plan_ids, pad])
                with torch.no_grad():
                    tok_embeds = embedding(plan_ids)           # (K, D)

                # Align soft_prompt from hidden-state space → input-embedding space
                # via W_a (LatentMAS §3.1).  This is the correct replacement for the
                # previous per-token norm scaling, which only fixed scale but left the
                # directional distribution OOD.  W_a is the closed-form linear map that
                # minimises the Wasserstein distance between the two distributions.
                with torch.no_grad():
                    sp = soft_prompt.to(device, dtype=torch.float32)
                    sp_scaled = (sp @ W_a).to(model_dtype)    # (K, D)

                # Query dropout decided BEFORE curriculum mixing so r can be
                # conditioned on whether the query is present. The dropout rate
                # is the probability of Q=0 (query dropped); L_sep only fires on
                # Q=0 steps, so higher dropout = more latent-grounding signal.
                # CLAUDE.md mandates 0.5; the May-14 weights at 0.1 starved L_sep.
                use_query = (random.random() >= query_dropout_rate)
                question = unit_question or PERSPECTIVE_PROMPTS[0]
                if use_query:
                    q_enc = base_tok(question, return_tensors="pt", truncation=True, max_length=32)
                    query_ids = q_enc["input_ids"].squeeze(0).long().to(device)  # (Q,)
                    with torch.no_grad():
                        query_embeds = embedding(query_ids).to(model_dtype)      # (Q, D)
                    Q = query_ids.shape[0]
                else:
                    query_embeds = torch.zeros(0, embedding.weight.shape[1],
                                               device=device, dtype=model_dtype)
                    Q = 0

                # Curriculum mixing.
                # r=0 always: inference always uses pure latents (r=0 in decode()),
                # so training must match. Random r was causing a distribution mismatch —
                # the model trained on 50%-token-embedded inputs but decoded with 100%
                # latent inputs. Per Interlat §3.2, the curriculum helps stability but
                # the inference distribution is pure-latent; we favour correctness here.
                r = 0.0
                mixed = CurriculumMixer.mix(
                    sp_scaled,
                    tok_embeds.to(model_dtype),
                    r,
                )                                              # (K, D)

                # Q=0: clip target to T=16.
                # T=3 was too sparse for the model to learn coherent generation —
                # with same-scale geometry (0.5B enc/dec) the model CAN exploit
                # the full latent signal, so we give it 16 tokens to train on.
                # T=16 still fits within the 64-token max_length cap on target_text.
                # plan_ids was already computed from full target_ids above.
                if Q == 0:
                    target_ids = target_ids[:16]

                # Frame: [bop, mixed (K), eop, query_toks (Q or 0), answer_toks (T)]
                with torch.no_grad():
                    tgt_embeds = embedding(target_ids).to(model_dtype)        # (T, D)
                A = asst_embed.shape[0]   # number of assistant-cue tokens (typically 3)

                # ── Phase B: prepend verbatim with p=0.5 ─────────────────
                # When hybrid_train is on and this bucket has cached verbatim,
                # prepend its embeddings to BOTH correct and wrong frames so the
                # LoRA learns to attend across [verbatim, soft_prompt, query].
                # Same verbatim used in both paths — the L_sep signal still
                # measures soft-prompt distinguishability holding context fixed.
                v_embeds = None
                V = 0
                if hybrid_train and bucket_id in bucket_verbatim_ids and random.random() < 0.5:
                    v_ids = bucket_verbatim_ids[bucket_id]
                    with torch.no_grad():
                        v_embeds = embedding(v_ids).to(model_dtype)
                    V = v_embeds.shape[0]

                parts: list = []
                if v_embeds is not None:
                    parts.append(v_embeds)
                parts.extend([bop_embed.unsqueeze(0), mixed, eop_embed.unsqueeze(0)])
                if Q > 0:
                    parts.append(query_embeds)
                parts.append(asst_embed)   # <|im_start|>assistant\n — matches decode()
                parts.append(tgt_embeds)
                inputs_embeds = torch.cat(parts)                               # (V+K+2+Q+A+T, D)

                wrong_id = random.choice([bid for bid in encoded_ids if bid != bucket_id])
                wrong_sp = bucket_soft[wrong_id].to(device, dtype=torch.float32)
                wrong_sp_scaled = (wrong_sp @ W_a).to(model_dtype)
                # Wrong path always uses r=0 (pure latents) so ALL K positions
                # differ from the correct path. When r is shared, high values
                # (e.g. 0.85) make 27/32 positions identical token embeddings —
                # the model can't distinguish correct from wrong at position 0.
                wrong_mixed = CurriculumMixer.mix(
                    wrong_sp_scaled,
                    tok_embeds.to(model_dtype),
                    0.0,
                )
                wrong_parts: list = []
                if v_embeds is not None:
                    wrong_parts.append(v_embeds)   # same verbatim as correct path
                wrong_parts.extend([bop_embed.unsqueeze(0), wrong_mixed, eop_embed.unsqueeze(0)])
                if Q > 0:
                    wrong_parts.append(query_embeds)
                wrong_parts.append(asst_embed)
                wrong_parts.append(tgt_embeds)
                inputs_embeds_wrong = torch.cat(wrong_parts)                  # (V+K+2+Q+A+T, D)

                # Plan path: query + target only (no latent frame) — used by L_align
                # to anchor the latent-conditioned distribution. Only built when the
                # query was not dropped (Q > 0); when dropped, L_align is skipped.
                batch: dict = {
                    "inputs_embeds":       inputs_embeds,
                    "inputs_embeds_wrong": inputs_embeds_wrong,
                    "target_ids":          target_ids,
                    "prefix_len":          V + K + 2 + Q + A,
                }
                if Q > 0:
                    inputs_embeds_plan = torch.cat([query_embeds, asst_embed, tgt_embeds], dim=0)
                    batch["inputs_embeds_plan"] = inputs_embeds_plan
                    batch["plan_prefix_len"] = Q + A

                is_last_in_epoch = (i == len(_epoch_units))
                should_step = (i % accum_steps == 0) or is_last_in_epoch
                losses = trainer.accumulate_step(batch, scale=accum_steps, step=should_step)
                all_task.append(losses["task"])
                all_sep.append(losses["sep"])
                (all_task_q0 if Q == 0 else all_task_q1).append(losses["task"])
                if Q == 0:
                    _epoch_task_q0.append(losses["task"])
                    _epoch_sep_q0.append(losses["sep"])
                else:
                    _epoch_task_q1.append(losses["task"])
                click.echo(
                    f"  Epoch {epoch+1} [{i}/{len(_epoch_units)}] bucket={bucket_id} "
                    f"task={losses['task']:.4f} sep={losses['sep']:.4f} "
                    + ("Q0 " if Q == 0 else "    ")
                    + (" ← OPT" if should_step else "")
                )
                if str(device).startswith("mps"):
                    torch.mps.empty_cache()

            except Exception as exc:
                click.echo(f"  Skipped {bucket_id}: {exc}", err=True)

        # Per-epoch summary — print after inner loop closes.
        _avg_q0  = sum(_epoch_task_q0) / max(1, len(_epoch_task_q0))
        _avg_q1  = sum(_epoch_task_q1) / max(1, len(_epoch_task_q1))
        _avg_sep = sum(_epoch_sep_q0)  / max(1, len(_epoch_sep_q0))
        _cur_lr  = trainer.optimizer.param_groups[0]["lr"]
        import statistics as _stats
        _median_sep_q0 = _stats.median(_epoch_sep_q0) if _epoch_sep_q0 else 0.0
        click.echo(
            f"── Epoch {epoch+1}/{epochs}  "
            f"task_q0={_avg_q0:.4f}  task_q1={_avg_q1:.4f}  "
            f"sep={_avg_sep:.6f}  median_sep_q0={_median_sep_q0:.6f}  "
            f"lr={_cur_lr:.2e}  "
            f"Q0={len(_epoch_task_q0)}  Q1={len(_epoch_task_q1)}"
        )

        # Sep watchdog: ONLY at end of epoch 3 (epoch == 2 in 0-indexed),
        # AND only if no good epoch has been recorded yet. The watchdog's
        # purpose is to bail early on "model totally ignores the latent" —
        # but if best-so-far already cleared 0.05, we have a usable
        # checkpoint and a transient dip at epoch 3 is just training noise
        # (margin saturation, lr cosine, dropout luck). In that case we keep
        # training; the best-checkpoint tracker preserves the high-water mark.
        # CLAUDE.md §LoRA rule.
        if (
            epoch == 2
            and _median_sep_q0 < 0.05
            and _avg_sep < 0.05
            and _best_mean_sep < 0.05
        ):
            raise click.ClickException(
                f"L_sep collapse: after 3 epoch(s), both "
                f"median_sep_q0={_median_sep_q0:.6f} and mean_sep_q0={_avg_sep:.6f} "
                f"are below the 0.05 threshold, AND no earlier epoch cleared it. "
                f"The receiver is genuinely ignoring the latent. Weights are NOT saved.\n\n"
                f"Investigate (per CLAUDE.md): "
                f"(a) is --query-dropout-rate >= 0.5? "
                f"(b) is inputs_embeds_wrong actually a different bucket? "
                f"(c) is _SEP_LAMBDA > 0 in lora_trainer.py? "
                f"(d) try --qa-per-bucket 5 + --epochs 10 for more sep-loaded steps."
            )

        # Best-checkpoint tracker — snapshot the LoRA state whenever this
        # epoch's mean sep exceeds the prior best. Cloned so subsequent epoch
        # updates don't mutate the snapshot.
        if _avg_sep > _best_mean_sep:
            _best_mean_sep = _avg_sep
            _best_lora_state = {
                k: v.detach().clone()
                for k, v in base_model.state_dict().items()
                if "lora_" in k
            }
            _best_epoch = epoch + 1
            click.echo(
                f"  ★ new best checkpoint  epoch={epoch+1}  "
                f"mean_sep={_avg_sep:.4f}  median_sep={_median_sep_q0:.4f}",
                err=True,
            )

    # Save the BEST checkpoint (highest mean sep) if we tracked one. Late
    # epochs may degrade due to margin saturation + lr decay; best-so-far
    # protects against shipping suboptimal final weights. Fall back to final
    # state if best tracking was somehow skipped.
    if _best_lora_state is not None:
        torch.save(_best_lora_state, bucket_dir / "lora_receiver.pt")
        click.echo(
            f"[libucks] Saved BEST LoRA checkpoint (epoch {_best_epoch}, "
            f"mean_sep={_best_mean_sep:.4f}) to lora_receiver.pt",
            err=True,
        )
    else:
        lora_state = {k: v for k, v in base_model.state_dict().items() if "lora_" in k}
        torch.save(lora_state, bucket_dir / "lora_receiver.pt")

    if all_task:
        n = min(5, len(all_task))
        q0_summary = (
            f"  task_q0 (latent-only): {sum(all_task_q0[:min(5,len(all_task_q0))])/max(1,min(5,len(all_task_q0))):.4f}"
            f" → {sum(all_task_q0[-min(5,len(all_task_q0)):])/max(1,min(5,len(all_task_q0))):.4f}"
            if all_task_q0 else "  task_q0: no Q=0 steps recorded"
        )
        q1_summary = (
            f"  task_q1 (query+latent): {sum(all_task_q1[:min(5,len(all_task_q1))])/max(1,min(5,len(all_task_q1))):.4f}"
            f" → {sum(all_task_q1[-min(5,len(all_task_q1)):])/max(1,min(5,len(all_task_q1))):.4f}"
            if all_task_q1 else "  task_q1: no Q>0 steps recorded"
        )
        click.echo(
            f"LoRA receiver training complete.\n"
            f"  task (all):  {sum(all_task[:n])/n:.4f} → {sum(all_task[-n:])/n:.4f}\n"
            f"  sep  (all):  {sum(all_sep[:n])/n:.4f} → {sum(all_sep[-n:])/n:.4f}\n"
            f"{q0_summary}\n"
            f"{q1_summary}\n"
            f"  Saved to {bucket_dir / 'lora_receiver.pt'}"
        )
    else:
        click.echo("No LoRA training steps completed.", err=True)


async def _train_creative(cfg, registry, store, bucket_ids, adapter, epochs, bucket_dir):
    """Creative mode: multi-perspective + hard negatives + InfoNCE."""
    from libucks.thinking import create_strategy
    from libucks.thinking.training.data_generator import MultiPerspectiveDataGenerator
    from libucks.thinking.training.train_adapter import ContrastiveAdapterTrainer

    latent_strategy = create_strategy(cfg)

    generator = MultiPerspectiveDataGenerator(
        latent_strategy=latent_strategy,
        registry=registry,
        store=store,
        teacher_model=cfg.model.anthropic_model,
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
    """Basic mode: per-slot cosine to bucket-token embeddings + diversity loss.

    L_task = mean over k of (1 - cos(adapter[k], interp(bucket_token_emb)[k]))
             — pulls each slot toward the k-th interpolated token-embedding of
               the bucket's source. K=32 outputs collectively summarize the
               bucket source as a sequence of K input embeddings.
             — On-manifold by construction: target IS embeddings.
             — Content-relevant: embeddings come from THIS bucket's tokens,
               not arbitrary vocab.
    L_div  = mean of (inter-slot cosine off-diagonal)^2
             — penalizes slot collapse.

    Replaces the earlier formulation that targeted encoder hidden states (off
    manifold, cos~0.14 to vocab) plus a bucket-agnostic L_manifold (pulled
    slots toward random vocab tokens, content-meaningless). The new target
    is content-relevant by construction — no separate manifold loss needed.
    """
    import random as _random
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from libucks.thinking import create_strategy
    from libucks.thinking.training.data_generator import (
        MultiPerspectiveDataGenerator, _collect_source_text,
    )

    latent_strategy = create_strategy(cfg)

    generator = MultiPerspectiveDataGenerator(
        latent_strategy=latent_strategy,
        registry=registry,
        store=store,
        teacher_model=cfg.model.anthropic_model,
    )
    optimizer = AdamW(adapter.parameters(), lr=1e-4)
    _device = next(adapter.parameters()).device
    K = adapter.output_len
    LAMBDA_DIV = 0.2   # reduced from 0.5: combined diversity pressure (div+xsep)
                       # at 0.8 overwhelmed L_task, driving outputs off-manifold
                       # (cos-to-vocab 0.475→0.145). At 0.2, total diversity
                       # pressure = 0.25 vs L_task = 1.0, giving manifold anchor
                       # room to win.

    click.echo(
        f"[libucks] _train_basic (B'): K={K}  λ_div={LAMBDA_DIV}  "
        f"target=bucket-token-embeddings",
        err=True,
    )

    vocab_emb: torch.Tensor | None = None
    tokenizer = None
    bucket_emb_cache: dict[str, torch.Tensor | None] = {}

    # Librarian-latent cache: skip the Anthropic teacher for buckets we've
    # already encoded once. First training run pays the API cost (3 calls per
    # bucket × 159 buckets ≈ $0.50); subsequent runs cost $0. Per-bucket file
    # so partial-failed runs preserve every paid call. Stored fp16 to halve disk.
    _lat_cache_dir = bucket_dir / "latent_cache"
    _lat_cache_dir.mkdir(exist_ok=True)
    _n_pre_cached = sum(1 for _ in _lat_cache_dir.glob("*.pt"))
    click.echo(
        f"[libucks] librarian-latent cache: {_n_pre_cached}/{len(bucket_ids)} buckets pre-cached "
        f"at {_lat_cache_dir} (delete to refetch)",
        err=True,
    )

    async def _get_librarian_latents(bid: str):
        """Return list of librarian-latent tensors for bid, from disk or teacher.

        Returns None if the bucket cannot be processed (API error, etc.).
        Cached tensors are stored fp16 on CPU; returned cast to fp32 on _device.
        """
        cache_path = _lat_cache_dir / f"{bid}.pt"
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                return [t.to(_device, torch.float32) for t in cached]
            except Exception as exc:
                click.echo(f"  Cache read failed for {bid} ({exc}); refetching", err=True)
        try:
            sample = await generator.generate(bid)
        except Exception as exc:
            click.echo(f"  Skipped {bid}: {exc}", err=True)
            return None
        cpu_fp16 = [t.detach().cpu().to(torch.float16) for t in sample.librarian_latents]
        try:
            torch.save(cpu_fp16, cache_path)
        except Exception as exc:
            click.echo(f"  Cache write failed for {bid} ({exc}); continuing without persist", err=True)
        return [t.to(_device, torch.float32) for t in cpu_fp16]

    # Preload encoder (the cache path may skip generator.generate, which used
    # to be the implicit trigger for model load).
    latent_strategy._mgr.get_model()

    def _bucket_token_emb(bucket_id: str) -> "torch.Tensor | None":
        """Return (T, base_dim) embeddings of the bucket's source tokens, or None."""
        if bucket_id in bucket_emb_cache:
            return bucket_emb_cache[bucket_id]
        try:
            front_matter, prose = store.read(bucket_id)
            # Match the teacher's source extraction: _collect_source_text walks
            # front_matter for the actual code body. Falling back only to prose
            # cost us ~30 buckets in the prior run (19% skip rate).
            source = (
                _collect_source_text(front_matter, max_chars=3000)
                or prose
                or getattr(front_matter, "domain_label", "")
                or ""
            )
            if len(source.strip()) < 10:
                bucket_emb_cache[bucket_id] = None
                return None
            tok = tokenizer(source, return_tensors="pt", truncation=True,
                            max_length=512, add_special_tokens=False)["input_ids"][0]
            if tok.shape[0] < 2:
                bucket_emb_cache[bucket_id] = None
                return None
            with torch.no_grad():
                emb = vocab_emb[tok.to(_device)]      # (T, base_dim)
            bucket_emb_cache[bucket_id] = emb
            return emb
        except Exception as exc:
            click.echo(f"  Failed to tokenize {bucket_id}: {exc}", err=True)
            bucket_emb_cache[bucket_id] = None
            return None

    for epoch in range(epochs):
        for i, bucket_id in enumerate(bucket_ids, 1):
            librarian_latents = await _get_librarian_latents(bucket_id)
            if librarian_latents is None:
                continue

            # Lazy-load vocab + tokenizer (encoder is preloaded above).
            if vocab_emb is None:
                _enc = latent_strategy._mgr.get_model()
                tokenizer = latent_strategy._mgr.get_tokenizer()
                with torch.no_grad():
                    vocab_emb = _enc.model.embed_tokens.weight.detach().to(
                        _device, torch.float32
                    )
                    if adapter.output_proj is not None:
                        vocab_emb = adapter.output_proj(vocab_emb)
                click.echo(
                    f"[libucks] vocab embeddings cached  shape={tuple(vocab_emb.shape)}",
                    err=True,
                )

            bt_emb = _bucket_token_emb(bucket_id)
            if bt_emb is None:
                click.echo(
                    f"  Skipped {bucket_id} L_task: insufficient source tokens", err=True,
                )
                continue

            latents = [t.clone() for t in librarian_latents]
            optimizer.zero_grad()
            output = adapter(latents)                   # (K, base_dim)

            # ── L_task: per-slot cos against interp(bucket_token_emb) ──────
            # bt_emb: (T, d) — already in base_dim
            with torch.no_grad():
                t_K = F.interpolate(
                    bt_emb.T.unsqueeze(0), size=K, mode="linear", align_corners=False,
                ).squeeze(0).T                              # (K, d)
                target_n = F.normalize(t_K, dim=-1)
            output_n = F.normalize(output, dim=-1)          # (K, d)
            L_task = (1.0 - (output_n * target_n).sum(dim=-1)).mean()

            # ── L_div: penalize slot collapse ──────────────────────────────
            inter_cos = (output_n @ output_n.T) - torch.eye(K, device=output.device)
            L_div = inter_cos.pow(2).mean()

            # ── L_xsep: cross-bucket cosine separation proxy ───────────────
            neg_id = _random.choice([b for b in bucket_ids if b != bucket_id])
            neg_latents = await _get_librarian_latents(neg_id)
            if neg_latents is not None:
                neg_out = adapter([t.clone() for t in neg_latents])
                pos_mean = F.normalize(output.mean(dim=0), dim=-1)
                neg_mean = F.normalize(neg_out.mean(dim=0), dim=-1)
                L_xsep = torch.dot(pos_mean, neg_mean).pow(2)
            else:
                L_xsep = torch.zeros((), device=output.device)

            LAMBDA_XSEP = 0.1
            loss = L_task + LAMBDA_DIV * L_div + LAMBDA_XSEP * L_xsep
            loss.backward()
            optimizer.step()
            click.echo(
                f"  Epoch {epoch+1} [{i}/{len(bucket_ids)}] "
                f"task={L_task.item():.4f} div={L_div.item():.4f} "
                f"xsep={L_xsep.item():.4f} total={loss.item():.4f}"
            )

    torch.save(adapter.state_dict(), bucket_dir / "adapter.pt")
    click.echo(f"Basic training complete. Saved to {bucket_dir / 'adapter.pt'}")


@cli.command("build-kv-cache")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--max-tokens", default=1024, show_default=True,
              help="Per-bucket max tokens encoded into the KV cache.")
def build_kv_cache_cmd(repo_path: Path | None, max_tokens: int):
    """Phase 4-C: precompute per-bucket KV caches for `cache_aug` inference.

    Iterates over registry buckets, runs Qwen 2.5-3B forward over each bucket's
    source text, saves the resulting past_key_values to
    <repo>/.libucks/kv_cache/<bucket_id>.safetensors. Required before
    `cache_aug_translator` can route to a bucket.
    """
    target = repo_path or _find_repo_root()
    asyncio.run(_run_build_kv_cache(target, max_tokens=max_tokens))


async def _run_build_kv_cache(repo_path: Path, *, max_tokens: int) -> None:
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from libucks.config import Config
    from libucks.storage.bucket_store import BucketStore
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.thinking.model_manager import ModelManager
    from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv
    from libucks.thinking.training.data_generator import _collect_source_text

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"
    registry = BucketRegistry(bucket_dir / "registry.json")
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        raise click.ClickException("no buckets — run `libucks init` first")

    # Cache aug receiver is locked to Qwen 2.5-3B (see train-cache-aug).
    receiver_model_id = "Qwen/Qwen2.5-3B"
    device = ModelManager._resolve_device(cfg.model.device)
    click.echo(f"[libucks:build-kv] loading {receiver_model_id} on {device}...", err=True)
    tokenizer = AutoTokenizer.from_pretrained(receiver_model_id)
    model = AutoModelForCausalLM.from_pretrained(receiver_model_id, dtype=torch.bfloat16)
    model.eval()
    model = model.to(device)

    kv_cache = BucketKVCache(bucket_dir / "kv_cache", model_id=receiver_model_id, max_tokens=max_tokens)

    built = 0
    skipped = 0
    t0 = time.time()
    for bid in bucket_ids:
        try:
            fm, prose = store.read(bid)
        except Exception as exc:
            click.echo(f"  {bid}: store.read failed ({exc}); skipping", err=True)
            skipped += 1
            continue
        chunks = list(fm.chunks)
        source_text = _collect_source_text(fm, max_chars=max_tokens * 4) or prose
        if not source_text:
            click.echo(f"  {bid}: empty source; skipping", err=True)
            skipped += 1
            continue
        flat = extract_bucket_kv(model, tokenizer, source_text, max_tokens=max_tokens)
        kv_cache.save(bid, flat, chunks)
        built += 1
    elapsed = time.time() - t0
    click.echo(
        f"[libucks:build-kv] built {built}/{len(bucket_ids)} (skipped {skipped}) "
        f"in {elapsed:.1f}s -> {bucket_dir / 'kv_cache'}",
        err=True,
    )


@cli.command("generate-qa")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--qa-per-bucket", "qa_per_bucket", default=5, show_default=True,
              help="Number of QA pairs to synthesize per bucket via the teacher.")
@click.option("--no-teacher", "no_teacher", is_flag=True, default=False,
              help="Skip the Anthropic teacher; fill missing buckets with source-text fallbacks only.")
def generate_qa_cmd(repo_path: Path | None, qa_per_bucket: int, no_teacher: bool):
    """Regenerate <repo>/.libucks/qa_cache.json by calling the Anthropic teacher.

    Does NOT retrain the Phase 4-A adapter+LoRA. Use this to refresh QA data
    independently of training (e.g. before `train-cache-aug` runs in Phase 4-C).
    """
    target = repo_path or _find_repo_root()
    if qa_per_bucket < 1 or qa_per_bucket > 10:
        raise click.ClickException(f"--qa-per-bucket must be in [1, 10]; got {qa_per_bucket}")
    asyncio.run(_regenerate_qa_cache(target, qa_per_bucket=qa_per_bucket, no_teacher=no_teacher))


async def _regenerate_qa_cache(repo_path: Path, *, qa_per_bucket: int, no_teacher: bool) -> None:
    """Run only the teacher Q&A generation portion of the training pipeline.

    Mirrors the logic in _train_lora_receiver's "Phase 1" block (the inline
    teacher fetch loop). Kept duplicated rather than refactored to minimise
    risk to the working Phase 4-A pipeline; can be DRY-ed later.
    """
    import os as _os
    from libucks.config import Config
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.thinking.training.data_generator import PERSPECTIVE_PROMPTS, _collect_source_text

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"
    registry = BucketRegistry(bucket_dir / "registry.json")
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        raise click.ClickException("no buckets — run `libucks init` first")

    teacher_client = None
    if not no_teacher:
        if not _os.environ.get("ANTHROPIC_API_KEY"):
            raise click.ClickException(
                "ANTHROPIC_API_KEY not set. Either set it in .env / shell, "
                "or pass --no-teacher (lower quality)."
            )
        try:
            import anthropic as _anthropic
            teacher_client = _anthropic.AsyncAnthropic()
        except ImportError:
            click.echo("[libucks] anthropic package not found — falling back to source-text", err=True)

    def _build_qa_prompt(n_pairs: int) -> str:
        count_word = "ONE question" if n_pairs <= 1 else f"{n_pairs} distinct questions"
        pair_word = "pair" if n_pairs <= 1 else "pairs"
        return (
            f"Given this source code, write {count_word} whose answers require specific "
            "facts from the code (function/class names, parameter signatures, constant "
            "values, return types, control flow). Each question must be ANSWERABLE only "
            "by reading this specific code — not by generic knowledge of the topic. "
            f"Make the {pair_word} cover different aspects of the code (different "
            "functions, branches, or invariants) — do not paraphrase the same question.\n\n"
            "Then write a concise 2-3 sentence plain English answer for each that "
            "explicitly names the relevant identifiers from the code.\n\n"
            "BAD example (too generic — answer comes from priors, not the code):\n"
            "QUESTION 1: How does this module handle errors?\n"
            "ANSWER 1: It catches exceptions and returns an error response.\n\n"
            "GOOD example (answer requires the code):\n"
            "QUESTION 1: What does load_config() return when the YAML file is missing?\n"
            "ANSWER 1: It returns the default Config() instance built from DEFAULT_SETTINGS, "
            "and logs a warning via logger.warning() rather than raising.\n\n"
            f"Format EXACTLY (numbered 1..{n_pairs}):\n"
            "QUESTION 1: <question>\nANSWER 1: <answer>\n"
            + ("QUESTION 2: <question>\nANSWER 2: <answer>\n..." if n_pairs > 1 else "")
        )

    def _parse_qa_pairs(text: str, n_expected: int, fallback_q: str, fallback_a: str):
        import re
        pattern = re.compile(
            r"QUESTION\s*(\d*)\s*:\s*(.+?)\s*ANSWER\s*\d*\s*:\s*(.+?)(?=\n\s*QUESTION\s*\d*\s*:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        pairs: list[tuple[str, str]] = []
        for m in pattern.finditer(text):
            q = m.group(2).strip()
            a = m.group(3).strip()
            if q and a:
                pairs.append((q, a))
            if len(pairs) >= n_expected:
                break
        if not pairs:
            return [(fallback_q, fallback_a)]
        return pairs

    _TEACHER_QA_PROMPT = _build_qa_prompt(qa_per_bucket)
    _qa_cache_path = bucket_dir / "qa_cache.json"
    _qa_cache: dict[str, dict] = {}
    if _qa_cache_path.exists():
        try:
            raw_cache = json.loads(_qa_cache_path.read_text())
            for k, v in raw_cache.items():
                if isinstance(v, dict) and "pairs" in v:
                    _qa_cache[k] = {
                        "pairs": [list(p) for p in v["pairs"]],
                        "source": v.get("source", ""),
                    }
        except Exception as _e:
            click.echo(f"[libucks] existing cache load failed ({_e}), starting fresh", err=True)
            _qa_cache = {}

    uncached = [
        bid for bid in bucket_ids
        if bid not in _qa_cache or len(_qa_cache[bid]["pairs"]) < qa_per_bucket
    ]
    click.echo(
        f"[libucks:generate-qa] {len(_qa_cache)} cached, {len(uncached)} to fetch "
        f"(target {qa_per_bucket} pairs/bucket)",
        err=True,
    )

    if teacher_client is not None and uncached:
        _sem = asyncio.Semaphore(1)
        import anthropic as _anthropic

        async def _fetch_qa(bucket_id: str):
            front_matter, prose = store.read(bucket_id)
            # max_chars=4096: _collect_source_text breaks (returns "") when the
            # first chunk exceeds max_chars instead of truncating it. Markdown
            # chunks routinely run 3-4KB; with max_chars=1024 the entire bucket
            # source falls back to its domain_label, ruining teacher Q&A on
            # ~25% of buckets. 4096 fits the typical first chunk.
            source_text = _collect_source_text(front_matter, max_chars=4096) or prose or front_matter.domain_label
            fallback_q = PERSPECTIVE_PROMPTS[0]
            fallback_a = source_text or ""
            pairs: list[tuple[str, str]] = [(fallback_q, fallback_a)]
            if source_text:
                try:
                    async with _sem:
                        max_toks = max(256, 200 * qa_per_bucket)
                        resp = await teacher_client.messages.create(
                            model=cfg.model.anthropic_model,
                            max_tokens=max_toks,
                            messages=[{"role": "user", "content": f"{_TEACHER_QA_PROMPT}\n\n{source_text}"}],
                        )
                        await asyncio.sleep(1.2)
                    raw = resp.content[0].text.strip()
                    pairs = _parse_qa_pairs(raw, qa_per_bucket, fallback_q, fallback_a)
                    click.echo(
                        f"  {bucket_id}: {len(pairs)} pairs  Q1={pairs[0][0][:50]}…",
                        err=True,
                    )
                except (_anthropic.AuthenticationError,
                        _anthropic.APIConnectionError,
                        _anthropic.RateLimitError) as fatal:
                    raise click.ClickException(f"Anthropic API fatal error: {fatal}") from fatal
                except Exception as transient:
                    click.echo(f"  {bucket_id}: teacher failed ({transient}) — fallback used", err=True)
            return bucket_id, pairs, source_text

        qa_results = await asyncio.gather(*[_fetch_qa(bid) for bid in uncached], return_exceptions=True)
        for item in qa_results:
            if isinstance(item, click.ClickException):
                raise item
            if isinstance(item, Exception):
                click.echo(f"  skipped (error): {item}", err=True)
                continue
            bid, pairs, src = item
            _qa_cache[bid] = {"pairs": [list(p) for p in pairs], "source": src or ""}
    elif uncached:
        for bucket_id in uncached:
            front_matter, prose = store.read(bucket_id)
            src = _collect_source_text(front_matter, max_chars=1024) or prose or front_matter.domain_label
            _qa_cache[bucket_id] = {
                "pairs": [[PERSPECTIVE_PROMPTS[0], src]],
                "source": src or "",
            }

    _qa_cache_path.write_text(json.dumps(_qa_cache, indent=2))
    total_pairs = sum(len(e["pairs"]) for e in _qa_cache.values())
    stubs = sum(
        1 for e in _qa_cache.values()
        for p in e["pairs"]
        if isinstance(p, list) and len(p) >= 1 and isinstance(p[0], str)
        and p[0].startswith("Explain concisely what this code does")
    )
    click.echo(
        f"[libucks:generate-qa] wrote {_qa_cache_path}: "
        f"{len(_qa_cache)} buckets, {total_pairs} total pairs, "
        f"{stubs} generic stubs ({100 * stubs / max(1, total_pairs):.1f}%)",
        err=True,
    )


@cli.command("train-cache-aug")
@click.option("--repo", "repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Path to repository (defaults to git repo containing cwd).")
@click.option("--epochs", default=3, show_default=True, help="Number of training epochs.")
@click.option("--lr", default=1e-4, show_default=True, help="AdamW base learning rate.")
@click.option("--warmup-steps", default=100, show_default=True,
              help="Linear warmup steps; lr ramps from 1/warmup_steps × base_lr up to base_lr.")
@click.option("--text-ratios", default="0.0,0.25,0.5,0.75,1.0", show_default=True,
              help="Comma-separated text_ratio choices for the Phase 4-C.5.5 curriculum. "
                   "Per sample, r ∼ uniform(choices); verbatim_chars = r × max-verbatim-chars.")
@click.option("--max-verbatim-chars", default=2400, show_default=True,
              help="Upper bound on verbatim prepended to the query when text_ratio=1.0.")
def train_cache_aug_cmd(
    repo_path: Path | None, epochs: int, lr: float, warmup_steps: int,
    text_ratios: str, max_verbatim_chars: int,
):
    """Phase 4-C.5/5.5: train Coprocessor + CrossBucketFusion against frozen Qwen 2.5-3B.

    Reads per-bucket KV caches from .libucks/kv_cache/, Q&A pairs from
    .libucks/qa_cache.json, and saves the trained coproc + fusion state_dict to
    .libucks/cache_aug_state.pt. Receiver stays frozen throughout.
    """
    target = repo_path or _find_repo_root()
    try:
        ratios = tuple(float(x.strip()) for x in text_ratios.split(",") if x.strip())
    except ValueError as exc:
        raise click.ClickException(f"--text-ratios must be comma-separated floats; got {text_ratios!r}") from exc
    if not ratios:
        ratios = (0.0,)
    asyncio.run(_run_train_cache_aug(
        target, epochs=epochs, lr=lr, warmup_steps=warmup_steps,
        text_ratios=ratios, max_verbatim_chars=max_verbatim_chars,
    ))


async def _run_train_cache_aug(
    repo_path: Path, *, epochs: int, lr: float, warmup_steps: int,
    text_ratios: tuple = (0.0,), max_verbatim_chars: int = 2400,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from libucks.config import Config
    from libucks.storage.bucket_store import BucketStore
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.thinking.model_manager import ModelManager
    from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
    from libucks.cache_augmentation.coprocessor import Coprocessor
    from libucks.cache_augmentation.fusion import CrossBucketFusion
    from libucks.thinking.training.cache_aug_trainer import CacheAugTrainer, load_qa_pairs

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"

    qa_path = bucket_dir / "qa_cache.json"
    if not qa_path.exists():
        raise click.ClickException(
            f"qa_cache.json missing at {qa_path}. Generate Q&A pairs via "
            "`libucks train-adapter --train-receiver` first."
        )
    samples = load_qa_pairs(qa_path)
    if not samples:
        raise click.ClickException(f"No usable Q&A samples in {qa_path}")

    registry = BucketRegistry(bucket_dir / "registry.json")
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_chunks = {}
    for bid in registry.get_all_centroids():
        fm, _ = store.read(bid)
        bucket_chunks[bid] = list(fm.chunks)

    # The cache-aug receiver is locked to Qwen 2.5-3B by the coprocessor's
    # architectural defaults (36 layers, 2 KV heads, head_dim=128, hidden=2048).
    # cfg.model.base_model is the Phase 4-A receiver knob and is not the
    # right field for this pipeline.
    receiver_model_id = "Qwen/Qwen2.5-3B"
    device = ModelManager._resolve_device(cfg.model.device)
    click.echo(f"[libucks:cache_aug] loading frozen {receiver_model_id} on {device}...", err=True)
    tokenizer = AutoTokenizer.from_pretrained(receiver_model_id)
    model = AutoModelForCausalLM.from_pretrained(receiver_model_id, dtype=torch.bfloat16)
    model.eval()
    model = model.to(device)

    coproc = Coprocessor().to(device).to(torch.float32)
    fusion = CrossBucketFusion().to(device).to(torch.float32)
    kv_cache = BucketKVCache(bucket_dir / "kv_cache", model_id=receiver_model_id)

    total_steps = max(1, len(samples) * epochs)
    trainer = CacheAugTrainer(
        base_model=model, tokenizer=tokenizer,
        coprocessor=coproc, fusion=fusion,
        bucket_kv_cache=kv_cache, bucket_chunks=bucket_chunks,
        lr=lr, warmup_steps=warmup_steps, total_steps=total_steps,
        store=store,
        text_ratio_choices=text_ratios,
        max_verbatim_chars=max_verbatim_chars,
    )

    click.echo(
        f"[libucks:cache_aug] training: {len(samples)} samples × {epochs} epochs "
        f"= {total_steps} steps, lr={lr:.2e}, warmup={warmup_steps}, "
        f"text_ratios={list(text_ratios)}, max_verbatim_chars={max_verbatim_chars}",
        err=True,
    )

    for epoch in range(epochs):
        stats = trainer.train_epoch(samples)
        click.echo(
            f"[libucks:cache_aug] epoch {epoch+1}/{epochs}: "
            f"mean_loss={stats['mean_loss']:.4f} n_steps={stats['n_steps']} "
            f"skipped={stats['skipped']}",
            err=True,
        )

    out_path = bucket_dir / "cache_aug_state.pt"
    torch.save({"coproc": coproc.state_dict(), "fusion": fusion.state_dict()}, out_path)
    click.echo(f"[libucks:cache_aug] saved {out_path}", err=True)


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

    from libucks.chunk_retriever import ChunkRetriever
    chunk_retriever = ChunkRetriever(
        cache_dir=bucket_dir / "chunk_emb_cache",
        embedder=embedder,
        store=store,
    )
    agent = CentralAgent(
        registry, cfg, embed_fn=embedder.embed,
        chunk_retriever=chunk_retriever,
    )
    librarians: dict[str, Librarian] = {}
    for bucket_id in bucket_ids:
        lib = Librarian(
            bucket_id=bucket_id,
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
            chunk_retriever=chunk_retriever,
        )
        librarians[bucket_id] = lib
        agent.register_librarian(bucket_id, lib)

    adapter = None
    if cfg.model.strategy == "latent":
        import torch
        from libucks.thinking.model_manager import ModelManager as _MM
        resolved_device = _MM._resolve_device(cfg.model.device)
        from transformers import AutoConfig as _AC2
        _base_hidden = _AC2.from_pretrained(cfg.model.base_model).hidden_size
        adapter = CommunicationAdapter(hidden_dim=strategy.hidden_dim, output_dim=_base_hidden,
                                       output_len=cfg.model.output_len)
        adapter.load_saved_weights(bucket_dir / "adapter.pt")

        # Load the Base receiver model first so we can read its actual dtype.
        click.echo("[libucks] loading Base receiver model for decode()...", err=True)
        strategy._mgr.load_base_model(
            model_id=cfg.model.base_model,
            quantization=cfg.model.quantization,
            bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
            device=cfg.model.device,
            base_model_dtype=cfg.model.base_model_dtype,
        )
        click.echo("[libucks] Base receiver model ready", err=True)

        # Cast adapter to match base model dtype — must happen after load_base_model()
        # so we read the real dtype rather than hardcoding float16.
        _base_model_dtype = strategy._mgr.get_base_model().dtype
        adapter = adapter.to(device=resolved_device, dtype=_base_model_dtype)

        _load_lora_weights(strategy, bucket_dir, resolved_device)

    translator = Translator(
        strategy, adapter=adapter, store=store,
        hybrid=cfg.model.hybrid_retrieval,
        verbatim_max_chars=cfg.model.hybrid_verbatim_max_chars,
        chunk_retriever=chunk_retriever,
    )

    orchestrator = QueryOrchestrator(
        central_agent=agent,
        librarians=librarians,
        embed_fn=embedder.embed,
        top_k=top_k,
    )

    click.echo(f"[libucks] routing: \"{query_text}\"", err=True)
    query_embedding = embedder.embed(query_text)
    pairs = await orchestrator.query(query_text)
    bucket_ids = [bid for bid, _ in pairs]
    representations = [rep for _, rep in pairs]
    click.echo(f"[libucks] {len(representations)} representations, synthesizing...", err=True)

    answer = await translator.synthesize(
        query_text, representations,
        bucket_ids=bucket_ids,
        query_embedding=query_embedding,
    )

    # Answer goes to stdout so it can be piped / captured cleanly.
    click.echo(answer)


@cli.command("hook")
@click.argument("event")
@click.argument("args", nargs=-1)
def hook_cmd(event: str, args: tuple):
    """Send a git hook event to the running libucks server (called by git hooks)."""
    import os as _os
    _env = _os.environ.get("LIBUCKS_REPO_PATH")
    repo_path = Path(_env).resolve() if _env else _find_repo_root()
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
