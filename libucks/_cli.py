"""CLI entry point — lives inside the package so the console script works from any directory."""
import asyncio
import json
import socket
import subprocess
from pathlib import Path

import click


def _load_lora_weights(strategy, bucket_dir: Path, device: str) -> None:
    """Inject LoRA structure into the Base receiver and load saved delta weights.

    Safe to call even if lora_receiver.pt does not exist — it silently skips.
    Must be called AFTER strategy._mgr.load_base_model().
    """
    lora_path = bucket_dir / "lora_receiver.pt"
    if not lora_path.exists():
        return
    import torch
    from libucks.thinking.training.lora_trainer import _inject_lora, _LORA_TARGETS
    base_model = strategy._mgr.get_base_model()
    _inject_lora(base_model, _LORA_TARGETS, r=4, alpha=4.0)
    state = torch.load(lora_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state, strict=False)
    click.echo("[libucks] LoRA receiver weights loaded", err=True)


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
@click.option("--no-teacher", "no_teacher", is_flag=True, default=False,
              help="Self-supervised mode: skip Anthropic teacher calls and encode bucket "
                   "prose directly via the local model. Useful when no API credits are available.")
@click.option("--train-receiver", "train_receiver", is_flag=True, default=False,
              help="Phase 2: after adapter training, fine-tune the Base receiver with "
                   "LoRA using L_task + L_sep (Interlat-Lite). Saves lora_receiver.pt.")
@click.option("--epochs", default=1, show_default=True, help="Number of training epochs.")
def train_adapter_cmd(
    repo_path: Path | None,
    creative: bool,
    no_teacher: bool,
    train_receiver: bool,
    epochs: int,
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
        train_receiver=train_receiver, epochs=epochs,
    ))


async def _run_train_adapter(
    repo_path: Path, creative: bool, no_teacher: bool, train_receiver: bool, epochs: int
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
    adapter = CommunicationAdapter(hidden_dim=_hidden_dim)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")

    from libucks.thinking.model_manager import ModelManager as _MM
    _training_device = _MM._resolve_device(cfg.model.device)
    adapter = adapter.to(_training_device)

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

    if train_receiver:
        click.echo(f"\n[libucks] Phase 2: LoRA receiver training on {len(bucket_ids)} buckets "
                   f"for {epochs} epoch(s)...")
        await _train_lora_receiver(cfg, store, bucket_ids, adapter, epochs, bucket_dir,
                                   no_teacher=no_teacher)


async def _train_no_teacher(cfg, store, bucket_ids, adapter, epochs, bucket_dir):
    """Self-supervised mode: encode bucket prose directly — no Anthropic teacher API needed.

    For each bucket:
      - Encode the prose with each of the three PERSPECTIVE_PROMPTS via reason().
      - Use the first encoding as the alignment target.
      - Train the adapter to map its own latent output toward that target (cosine loss).

    This verifies the full forward-pass pipeline without requiring API credits.
    """
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from libucks.thinking import create_strategy
    from libucks.thinking.training.data_generator import PERSPECTIVE_PROMPTS

    latent_strategy = create_strategy(cfg)
    optimizer = AdamW(adapter.parameters(), lr=1e-4)
    _device = next(adapter.parameters()).device

    all_losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, bucket_id in enumerate(bucket_ids, 1):
            try:
                _, prose = store.read(bucket_id)

                # Encode prose from three perspectives (no API call needed)
                latents: list[torch.Tensor] = []
                for prompt in PERSPECTIVE_PROMPTS:
                    hidden = await latent_strategy.reason(prompt, prose)
                    latents.append(hidden.clone().detach().to(_device, torch.float32))

                # Self-supervised target: first perspective encoding
                target_raw = latents[0].mean(dim=0)
                target = F.normalize(target_raw, dim=0)

                optimizer.zero_grad()
                output = adapter(latents)
                anchor = F.normalize(output.mean(dim=0), dim=0)
                loss = 1.0 - torch.dot(anchor, target)
                loss.backward()
                optimizer.step()

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
                               no_teacher: bool = False):
    """Phase 2 (Interlat-Lite): fine-tune the Base receiver with LoRA.

    Loss: L_total = L_task - λ_sep * L_sep  (λ_sep = 0.1)

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
    K = adapter.output_len  # 32

    # Build <bop> / <eop> boundary embeddings (mirrors LatentStrategy.decode())
    bop_id = base_tok.convert_tokens_to_ids("<|im_start|>")
    eop_id = base_tok.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        # Cast to float32: on MPS the embedding layer is float16, but mixed, tgt_embeds
        # and sp_scaled are all float32.  torch.cat is strict about dtype — a float16
        # boundary embedding in the same cat as float32 mixed latents raises
        # RuntimeError and silently skips every bucket.
        bop_embed = embedding(torch.tensor([bop_id], device=device)).squeeze(0).detach().float()
        eop_embed = embedding(torch.tensor([eop_id], device=device)).squeeze(0).detach().float()

    # Initialise LoRA receiver trainer (injects LoRA in-place on base_model)
    trainer = LoRAReceiverTrainer(base_model, lora_r=4, lora_alpha=4.0, lr=2e-4)

    # Pre-encode all buckets so we don't call the model twice per step.
    # The latent encoding always uses the raw source text as the context
    # for reason() — that's the semantic content being compressed.
    # The CE training TARGET is either a teacher-generated English description
    # (default) or the raw source text (--no-teacher fallback).
    from libucks.thinking.training.data_generator import _collect_source_text

    # Instantiate the Anthropic teacher client once (reads ANTHROPIC_API_KEY
    # from the environment, same as the rest of the codebase).
    teacher_client = None
    if not no_teacher:
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
        "Given this source code, write one question a developer might ask about it, "
        "then write a concise 2-3 sentence plain English answer. "
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

    click.echo("[libucks] pre-encoding bucket latents...", err=True)
    bucket_soft: dict[str, torch.Tensor] = {}
    bucket_target: dict[str, str] = {}
    bucket_query: dict[str, str] = {}   # question text used as query conditioning
    for bucket_id in bucket_ids:
        try:
            front_matter, prose = store.read(bucket_id)
            # Source text drives the latent encoding (what the adapter compresses).
            source_text = _collect_source_text(front_matter, max_chars=1024) or prose or front_matter.domain_label

            # Default fallbacks (--no-teacher or API unavailable)
            question = PERSPECTIVE_PROMPTS[0]
            target_text = source_text

            if teacher_client is not None and source_text:
                try:
                    resp = await teacher_client.messages.create(
                        model=cfg.model.anthropic_model,
                        max_tokens=192,
                        messages=[{"role": "user", "content": f"{_TEACHER_QA_PROMPT}\n\n{source_text}"}],
                    )
                    raw = resp.content[0].text.strip()
                    question, target_text = _parse_qa(raw, question, source_text)
                    click.echo(
                        f"  teacher Q&A for {bucket_id}:\n"
                        f"    Q: {question[:80]}\n"
                        f"    A: {target_text[:80]}...",
                        err=True,
                    )
                except (_anthropic.AuthenticationError,
                        _anthropic.APIConnectionError,
                        _anthropic.RateLimitError) as fatal_exc:
                    # Configuration/quota errors — abort immediately so the user
                    # never silently trains on the wrong target distribution.
                    raise click.ClickException(
                        f"Anthropic API fatal error during LoRA receiver training — "
                        f"check ANTHROPIC_API_KEY and credit balance: {fatal_exc}"
                    ) from fatal_exc
                except Exception as transient_exc:
                    click.echo(
                        f"  Teacher call failed (transient) for {bucket_id}: "
                        f"{transient_exc} — using source text",
                        err=True,
                    )
                    # question and target_text remain at their fallback values

            bucket_target[bucket_id] = target_text
            bucket_query[bucket_id] = question
            hidden = await latent_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
            with torch.no_grad():
                soft = adapter([hidden.clone().detach().to(device, torch.float32)])
            bucket_soft[bucket_id] = soft.detach()
            click.echo(f"  encoded {bucket_id} → soft-prompt {tuple(soft.shape)}", err=True)
        except click.ClickException:
            raise  # propagate fatal errors
        except Exception as exc:
            click.echo(f"  Skipped encoding {bucket_id}: {exc}", err=True)

    if not bucket_soft:
        click.echo("No buckets encoded — aborting LoRA training.", err=True)
        return

    encoded_ids = list(bucket_soft.keys())
    all_task: list[float] = []
    all_sep: list[float] = []

    for epoch in range(epochs):
        for i, bucket_id in enumerate(encoded_ids, 1):
            try:
                soft_prompt = bucket_soft[bucket_id]          # (K, D)
                target_text = bucket_target[bucket_id]

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

                # Scale-normalise soft_prompt to match the Base model's native
                # embedding scale before mixing and framing.  The adapter was
                # trained against Instruct hidden states (scale ~10-50/dim) but
                # the Base model's embed_tokens layer has scale ~2-3/dim.  The
                # mismatch causes attention Q×K products to overflow → NaN.
                # We rescale each token vector individually (per-token L2 norm)
                # to the median norm of the vocabulary embeddings.
                with torch.no_grad():
                    embed_norm = embedding.weight.data.norm(dim=-1).median()
                    sp = soft_prompt.to(device, torch.float32)
                    sp_norms = sp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    sp_scaled = sp / sp_norms * embed_norm    # (K, D) at embed scale

                # Curriculum mixing
                r = random.uniform(0.0, 1.0)
                mixed = CurriculumMixer.mix(
                    sp_scaled,
                    tok_embeds.to(torch.float32),
                    r,
                )                                              # (K, D)

                # Tokenise and embed the question (max 32 tokens)
                question = bucket_query.get(bucket_id, PERSPECTIVE_PROMPTS[0])
                q_enc = base_tok(question, return_tensors="pt", truncation=True, max_length=32)
                query_ids = q_enc["input_ids"].squeeze(0).long().to(device)  # (Q,)
                with torch.no_grad():
                    query_embeds = embedding(query_ids).to(torch.float32)    # (Q, D)
                Q = query_ids.shape[0]

                # Frame: [bop, mixed (K), eop, query_toks (Q), answer_toks (T)]
                with torch.no_grad():
                    tgt_embeds = embedding(target_ids)         # (T, D)
                inputs_embeds = torch.cat(
                    [bop_embed.unsqueeze(0), mixed, eop_embed.unsqueeze(0),
                     query_embeds, tgt_embeds.to(torch.float32)]
                )                                              # (K+2+Q+T, D)

                # Wrong-path: pick a different bucket (cycle if only one bucket)
                wrong_id = next(
                    (bid for bid in encoded_ids if bid != bucket_id),
                    bucket_id,
                )
                wrong_sp = bucket_soft[wrong_id].to(device, torch.float32)
                wrong_sp_norms = wrong_sp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                wrong_sp_scaled = wrong_sp / wrong_sp_norms * embed_norm
                wrong_mixed = CurriculumMixer.mix(
                    wrong_sp_scaled,
                    tok_embeds.to(torch.float32),
                    r,
                )
                inputs_embeds_wrong = torch.cat(
                    [bop_embed.unsqueeze(0), wrong_mixed, eop_embed.unsqueeze(0),
                     query_embeds, tgt_embeds.to(torch.float32)]
                )                                              # (K+2+Q+T, D)

                batch = {
                    "inputs_embeds":       inputs_embeds,
                    "inputs_embeds_wrong": inputs_embeds_wrong,
                    "target_ids":          target_ids,
                    "prefix_len":          K + 2 + Q,
                }

                losses = trainer.train_step(batch)
                all_task.append(losses["task"])
                all_sep.append(losses["sep"])
                click.echo(
                    f"  Epoch {epoch+1} [{i}/{len(encoded_ids)}] bucket={bucket_id} "
                    f"task={losses['task']:.4f} sep={losses['sep']:.4f}"
                )

            except Exception as exc:
                click.echo(f"  Skipped {bucket_id}: {exc}", err=True)

    # Save only LoRA delta weights (lora_A / lora_B keys)
    lora_state = {k: v for k, v in base_model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, bucket_dir / "lora_receiver.pt")

    if all_task:
        n = min(5, len(all_task))
        click.echo(
            f"LoRA receiver training complete. "
            f"task: {sum(all_task[:n])/n:.4f} → {sum(all_task[-n:])/n:.4f}  "
            f"sep: {sum(all_sep[:n])/n:.4f} → {sum(all_sep[-n:])/n:.4f}. "
            f"Saved to {bucket_dir / 'lora_receiver.pt'}"
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
    """Basic mode: MSE between adapter mean-pooled output and target latent."""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from libucks.thinking import create_strategy
    from libucks.thinking.training.data_generator import MultiPerspectiveDataGenerator

    latent_strategy = create_strategy(cfg)

    generator = MultiPerspectiveDataGenerator(
        latent_strategy=latent_strategy,
        registry=registry,
        store=store,
        teacher_model=cfg.model.anthropic_model,
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
        from libucks.thinking.model_manager import ModelManager as _MM
        resolved_device = _MM._resolve_device(cfg.model.device)
        adapter = CommunicationAdapter(hidden_dim=strategy.hidden_dim)
        adapter.load_saved_weights(bucket_dir / "adapter.pt")
        # dtype must match the model's output dtype. ModelManager loads Qwen in
        # float16 on MPS; float32 adapter parameters cause an MPS broadcast error.
        adapter_dtype = torch.float16 if resolved_device == "mps" else None
        adapter = adapter.to(device=resolved_device, dtype=adapter_dtype)

        # Load the Base receiver model required by LatentStrategy.decode()
        click.echo("[libucks] loading Base receiver model for decode()...", err=True)
        strategy._mgr.load_base_model(
            model_id=cfg.model.base_model,
            quantization=cfg.model.quantization,
            bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
            device=cfg.model.device,
        )
        click.echo("[libucks] Base receiver model ready", err=True)

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
