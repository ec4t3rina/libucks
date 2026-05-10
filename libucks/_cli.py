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
def train_adapter_cmd(
    repo_path: Path | None,
    creative: bool,
    no_teacher: bool,
    train_receiver: bool,
    receiver_only: bool,
    epochs: int,
    accum_steps: int,
    lora_lr: float,
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
    asyncio.run(_run_train_adapter(
        target, creative=creative, no_teacher=no_teacher,
        train_receiver=train_receiver, receiver_only=receiver_only, epochs=epochs,
        accum_steps=accum_steps, lora_lr=lora_lr,
    ))


async def _run_train_adapter(
    repo_path: Path, creative: bool, no_teacher: bool, train_receiver: bool,
    receiver_only: bool = False, epochs: int = 1, accum_steps: int = 8, lora_lr: float = 1e-4,
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
    adapter = CommunicationAdapter(hidden_dim=_hidden_dim, output_dim=_base_dim)
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
                                   accum_steps=accum_steps, lora_lr=lora_lr)


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
                               lora_lr: float = 1e-4):
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

    # Q&A prompt: the teacher generates a question + answer pair so the LoRA
    # receiver learns to answer specific questions, not just describe code.
    # The question becomes the query conditioning prefix; the answer is the CE target.
    _TEACHER_QA_PROMPT = (
        "Given this source code, write ONE question whose answer requires specific "
        "facts from the code (function/class names, parameter signatures, constant "
        "values, return types, control flow). The question must be ANSWERABLE only "
        "by reading this specific code — not by generic knowledge of the topic.\n\n"
        "Then write a concise 2-3 sentence plain English answer that explicitly "
        "names the relevant identifiers from the code.\n\n"
        "BAD example (too generic — answer comes from priors, not the code):\n"
        "QUESTION: How does this module handle errors?\n"
        "ANSWER: It catches exceptions and returns an error response to the caller.\n\n"
        "GOOD example (answer requires the code):\n"
        "QUESTION: What does load_config() return when the YAML file is missing?\n"
        "ANSWER: It returns the default Config() instance built from DEFAULT_SETTINGS, "
        "and logs a warning via logger.warning() rather than raising. The Path.exists() "
        "check at the top of the function gates this fallback.\n\n"
        "Format EXACTLY as:\nQUESTION: <question>\nANSWER: <answer>"
    )

    def _parse_qa(text: str, fallback_q: str, fallback_a: str):
        """Return (question, answer) parsed from QUESTION:/ANSWER: format."""
        q = a = None
        for line in text.splitlines():
            if line.startswith("QUESTION:"):
                q = line[len("QUESTION:"):].strip()
            elif line.startswith("ANSWER:"):
                a = line[len("ANSWER:"):].strip()
        # ANSWER: may span multiple lines — grab everything after the marker
        if a is None and "ANSWER:" in text:
            a = text.split("ANSWER:", 1)[1].strip()
        return (q or fallback_q), (a or fallback_a)

    # ── Phase 1: parallel teacher API calls (I/O-bound, up to 5 concurrent) ──
    _qa_cache_path = bucket_dir / "qa_cache.json"
    _qa_cache: dict[str, tuple[str, str, str]] = {}  # bucket_id → (question, target, source)

    # Load persisted cache — allows re-runs after Phase 2 failures without re-fetching.
    if _qa_cache_path.exists():
        try:
            raw_cache = json.loads(_qa_cache_path.read_text())
            _qa_cache = {k: tuple(v) for k, v in raw_cache.items()}  # type: ignore[assignment]
            click.echo(f"[libucks] Loaded {len(_qa_cache)} Q&A pairs from cache", err=True)
        except Exception as _e:
            click.echo(f"[libucks] Cache load failed ({_e}), re-fetching", err=True)
            _qa_cache = {}

    uncached = [bid for bid in bucket_ids if bid not in _qa_cache]
    click.echo(
        f"[libucks] Phase 1: {len(_qa_cache)} cached, {len(uncached)} to fetch...", err=True
    )

    if teacher_client is not None and uncached:
        _sem = asyncio.Semaphore(1)

        async def _fetch_qa(bucket_id: str) -> tuple[str, str, str, str | None]:
            front_matter, prose = store.read(bucket_id)
            source_text = _collect_source_text(front_matter, max_chars=1024) or prose or front_matter.domain_label
            question = PERSPECTIVE_PROMPTS[0]
            target_text = source_text
            if source_text:
                try:
                    async with _sem:
                        resp = await teacher_client.messages.create(
                            model=cfg.model.anthropic_model,
                            max_tokens=256,
                            messages=[{"role": "user", "content": f"{_TEACHER_QA_PROMPT}\n\n{source_text}"}],
                        )
                        await asyncio.sleep(1.2)  # stay under 50 RPM limit
                    raw = resp.content[0].text.strip()
                    question, target_text = _parse_qa(raw, question, source_text)
                    click.echo(
                        f"  Q&A {bucket_id}: Q={question[:60]} | A={target_text[:60]}...",
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
            return bucket_id, question, target_text, source_text

        qa_results = await asyncio.gather(*[_fetch_qa(bid) for bid in uncached],
                                          return_exceptions=True)
        for item in qa_results:
            if isinstance(item, click.ClickException):
                raise item
            if isinstance(item, Exception):
                click.echo(f"  Skipped bucket (API error): {item}", err=True)
                continue
            bid, q, t, src = item
            _qa_cache[bid] = (q, t, src)

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
            _qa_cache[bucket_id] = (PERSPECTIVE_PROMPTS[0], src, src)

    click.echo(f"[libucks] Phase 1 done — {len(_qa_cache)} Q&A pairs collected", err=True)

    # ── Phase 2: sequential GPU encoding (serialized by _device_lock on MPS) ──
    click.echo("[libucks] Phase 2: encoding latents sequentially...", err=True)
    bucket_soft: dict[str, torch.Tensor] = {}
    bucket_target: dict[str, str] = {}
    bucket_query: dict[str, str] = {}

    for bucket_id in bucket_ids:
        if bucket_id not in _qa_cache:
            continue
        question, target_text, source_text = _qa_cache[bucket_id]
        try:
            hidden = await latent_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
            with torch.no_grad():
                soft = adapter([hidden.clone().detach().to(device, model_dtype)])
            bucket_soft[bucket_id] = soft.detach()
            bucket_target[bucket_id] = target_text
            bucket_query[bucket_id] = question
            click.echo(f"  encoded {bucket_id} → soft-prompt {tuple(soft.shape)}", err=True)
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

    # Pre-flight data quality check — surface degenerate targets before wasting compute.
    _usable = [
        bid for bid in encoded_ids
        if not bucket_target[bid].startswith("# /") and " " in bucket_target[bid].strip()
    ]
    if len(_usable) < len(encoded_ids):
        _skipped = len(encoded_ids) - len(_usable)
        _pct = 100 * _skipped // len(encoded_ids)
        click.echo(
            f"[libucks] WARNING: {_skipped}/{len(encoded_ids)} buckets ({_pct}%) have "
            f"degenerate targets (raw source dumps or bare identifiers — teacher API "
            f"fallback). Delete .libucks/qa_cache.json and re-run without --no-teacher "
            f"to regenerate. Training on {len(_usable)} buckets only.",
            err=True,
        )
        if len(_usable) == 0:
            raise click.ClickException(
                "All buckets have degenerate targets. Cannot train. "
                "Delete .libucks/qa_cache.json and re-run without --no-teacher."
            )
        encoded_ids = _usable

    # Initialise LoRA receiver trainer now that encoded_ids is final.
    # Warmup: 5% of total steps (floor 5). 20% was too long — wasted the first epoch
    # at near-zero lr when we need every step to count on small datasets.
    _total_opt_steps = max(1, (epochs * len(encoded_ids)) // accum_steps)
    _warmup_steps = max(5, _total_opt_steps // 20)
    trainer = LoRAReceiverTrainer(
        base_model, lora_r=16, lora_alpha=16.0, lr=lora_lr,
        warmup_steps=_warmup_steps, total_steps=_total_opt_steps,
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

    all_task: list[float] = []
    all_sep: list[float] = []
    all_task_q0: list[float] = []   # task loss on query-dropped steps (latent-only)
    all_task_q1: list[float] = []   # task loss on query-present steps

    for epoch in range(epochs):
        _epoch_task_q0: list[float] = []
        _epoch_sep_q0:  list[float] = []
        _epoch_task_q1: list[float] = []
        for i, bucket_id in enumerate(encoded_ids, 1):
            try:
                soft_prompt = bucket_soft[bucket_id]          # (K, D)
                target_text = bucket_target[bucket_id]

                # Skip degenerate targets: raw source dumps (teacher API fallback)
                # and bare identifiers (teacher echoed back the bucket name).
                # These produce CE loss of 10–16 nats and poison the gradient.
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
                # conditioned on whether the query is present.
                # 90/10 Q=1 / Q=0. With _SEP_LAMBDA=0 and _ALIGN_LAMBDA=0, Q=0 steps
                # only contribute CE on a no-query prefix — same loss path, different
                # input distribution from inference. Heavy Q=1 dose pulls the LoRA
                # toward the actual inference distribution; the 10% Q=0 retains a
                # task_q0 measurement so we can detect if the model abandons
                # latent-only decoding entirely.
                use_query = (random.random() >= 0.1)  # 90% Q=1, 10% Q=0
                question = bucket_query.get(bucket_id, PERSPECTIVE_PROMPTS[0])
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
                parts = [bop_embed.unsqueeze(0), mixed, eop_embed.unsqueeze(0)]
                if Q > 0:
                    parts.append(query_embeds)
                parts.append(asst_embed)   # <|im_start|>assistant\n — matches decode()
                parts.append(tgt_embeds)
                inputs_embeds = torch.cat(parts)                               # (K+2+Q+A+T, D)

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
                wrong_parts = [bop_embed.unsqueeze(0), wrong_mixed, eop_embed.unsqueeze(0)]
                if Q > 0:
                    wrong_parts.append(query_embeds)
                wrong_parts.append(asst_embed)
                wrong_parts.append(tgt_embeds)
                inputs_embeds_wrong = torch.cat(wrong_parts)                  # (K+2+Q+A+T, D)

                # Plan path: query + target only (no latent frame) — used by L_align
                # to anchor the latent-conditioned distribution. Only built when the
                # query was not dropped (Q > 0); when dropped, L_align is skipped.
                batch: dict = {
                    "inputs_embeds":       inputs_embeds,
                    "inputs_embeds_wrong": inputs_embeds_wrong,
                    "target_ids":          target_ids,
                    "prefix_len":          K + 2 + Q + A,
                }
                if Q > 0:
                    inputs_embeds_plan = torch.cat([query_embeds, asst_embed, tgt_embeds], dim=0)
                    batch["inputs_embeds_plan"] = inputs_embeds_plan
                    batch["plan_prefix_len"] = Q + A

                is_last_in_epoch = (i == len(encoded_ids))
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
                    f"  Epoch {epoch+1} [{i}/{len(encoded_ids)}] bucket={bucket_id} "
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
        click.echo(
            f"── Epoch {epoch+1}/{epochs}  "
            f"task_q0={_avg_q0:.4f}  task_q1={_avg_q1:.4f}  "
            f"sep={_avg_sep:.6f}  "
            f"lr={_cur_lr:.2e}  "
            f"Q0={len(_epoch_task_q0)}  Q1={len(_epoch_task_q1)}"
        )

    # Save only LoRA delta weights (lora_A / lora_B keys)
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
        from libucks.thinking.model_manager import ModelManager as _MM
        resolved_device = _MM._resolve_device(cfg.model.device)
        from transformers import AutoConfig as _AC2
        _base_hidden = _AC2.from_pretrained(cfg.model.base_model).hidden_size
        adapter = CommunicationAdapter(hidden_dim=strategy.hidden_dim, output_dim=_base_hidden)
        adapter.load_saved_weights(bucket_dir / "adapter.pt")

        # Load the Base receiver model first so we can read its actual dtype.
        click.echo("[libucks] loading Base receiver model for decode()...", err=True)
        strategy._mgr.load_base_model(
            model_id=cfg.model.base_model,
            quantization=cfg.model.quantization,
            bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
            device=cfg.model.device,
        )
        click.echo("[libucks] Base receiver model ready", err=True)

        # Cast adapter to match base model dtype — must happen after load_base_model()
        # so we read the real dtype rather than hardcoding float16.
        _base_model_dtype = strategy._mgr.get_base_model().dtype
        adapter = adapter.to(device=resolved_device, dtype=_base_model_dtype)

        _load_lora_weights(strategy, bucket_dir, resolved_device)

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
