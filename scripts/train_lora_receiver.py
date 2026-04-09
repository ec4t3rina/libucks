
import asyncio
import argparse
import gc
import re
from pathlib import Path


async def main(repo_path: Path, epochs: int, device: str, output: Path) -> None:
    import torch
    from libucks.config import Config
    from libucks.thinking.model_manager import ModelManager
    from libucks.thinking.latent_strategy import LatentStrategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.thinking.training.data_generator import (
        PERSPECTIVE_PROMPTS, _collect_source_text,
    )
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer
    from libucks.thinking.text_strategy import TextStrategy
    from libucks.storage.bucket_store import BucketStore
    from libucks.storage.bucket_registry import BucketRegistry

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"

    # ── Task 4: Remove stale weights to prevent dim mismatch (896 vs 2048) ───
    for stale in [bucket_dir / "adapter.pt", bucket_dir / "base_receiver_lora.pt"]:
        if stale.exists():
            print(f"[train] Removing stale weights: {stale}")
            stale.unlink()

    # ── Load buckets ──────────────────────────────────────────────────────────
    registry = BucketRegistry(repo_path / cfg.paths.registry_file)
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        print("[train] ERROR: No buckets found. Run `libucks init` first.")
        return
    print(f"[train] {len(bucket_ids)} buckets found")

    text_strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    # ── PHASE A: Encoder only — pre-compute all latents ──────────────────────
    # Both 3B models at once (even 4-bit) exhaust 16 GB. Encode everything while
    # only the encoder is in memory, store tiny latents on CPU, then unload.
    print(f"[train] Phase A: Loading Instruct encoder on {device}...")
    mgr = ModelManager()
    mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="4bit", device=device)
    strategy = LatentStrategy(mgr)

    print("[train] Pre-computing latents for all buckets...")
    correct_latents: dict = {}   # bucket_id → (seq_len, D) tensor on CPU
    wrong_latents: dict = {}     # bucket_id → (seq_len, D) tensor on CPU
    target_texts: dict = {}      # bucket_id → teacher summary string

    for i, bucket_id in enumerate(bucket_ids, 1):
        print(f"  [{i}/{len(bucket_ids)}] Encoding {bucket_id}...")
        front_matter, prose = store.read(bucket_id)
        source_text = _collect_source_text(front_matter, max_chars=3000) or prose

        summary_text = await text_strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
        # Strip markdown bold/italic markers so the model learns plain prose
        # rather than overfitting to ** token patterns in the training summaries.
        summary_text = re.sub(r'\*+', '', summary_text).strip()
        target_texts[bucket_id] = summary_text

        latent = await strategy.reason(PERSPECTIVE_PROMPTS[0], source_text)
        correct_latents[bucket_id] = latent.cpu()
        del latent

        wrong_id = next(bid for bid in bucket_ids if bid != bucket_id)
        _, wrong_prose = store.read(wrong_id)
        wrong_latent = await strategy.encode(wrong_prose)
        wrong_latents[bucket_id] = wrong_latent.cpu()
        del wrong_latent

    # Unload encoder — free ~6 GB (float16) or ~2 GB (true 4-bit NF4)
    print("[train] Unloading encoder...")
    mgr.unload_encoder()
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ── PHASE B: Receiver only ────────────────────────────────────────────────
    print(f"[train] Phase B: Loading Base receiver on {device}...")
    mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="4bit", device=device)
    base_model = mgr.get_base_model()
    base_tokenizer = mgr.get_base_tokenizer()
    embedding_layer = base_model.model.embed_tokens

    hidden_dim = base_model.config.hidden_size
    adapter = CommunicationAdapter(hidden_dim=hidden_dim)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")
    if device == "mps":
        adapter = adapter.to(device=device, dtype=torch.float16)
    else:
        adapter = adapter.to(device=device)

    total_steps = epochs * len(bucket_ids)
    warmup_steps = max(1, total_steps // 10)
    trainer = LoRAReceiverTrainer(base_model, lora_r=4, lora_alpha=4, lr=1e-4, warmup_steps=warmup_steps)

    # Token recycling: reuse Qwen's native chat-boundary tokens as frame markers.
    # No add_special_tokens / resize_token_embeddings needed — keeps embed_tokens
    # and lm_head fully frozen, avoiding ~5 GB of AdamW optimizer state.
    bop_id = base_tokenizer.convert_tokens_to_ids("<|im_start|>")
    eop_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    bop_embed = embedding_layer(
        torch.tensor([bop_id], device=device)
    ).squeeze(0).detach()
    eop_embed = embedding_layer(
        torch.tensor([eop_id], device=device)
    ).squeeze(0).detach()

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"[train] Starting {epochs} epoch(s) of LoRA training...")
    for epoch in range(epochs):
        epoch_task_loss = 0.0
        epoch_sep_loss = 0.0
        count = 0

        for i, bucket_id in enumerate(bucket_ids, 1):
            print(f"  Epoch {epoch+1}/{epochs}  bucket {i}/{len(bucket_ids)}: {bucket_id}")
            try:
                # Build correct-path inputs_embeds using pre-computed latent
                correct_lat = correct_latents[bucket_id].to(device)
                soft_prompt = adapter([correct_lat])                             # (K, D)
                framed = adapter.frame(soft_prompt, bop_embed, eop_embed)       # (K+2, D)
                del correct_lat, soft_prompt

                # Build wrong-path inputs_embeds
                wrong_lat = wrong_latents[bucket_id].to(device)
                wrong_soft = adapter([wrong_lat])                                # (K, D)
                framed_wrong = adapter.frame(wrong_soft, bop_embed, eop_embed)  # (K+2, D)
                del wrong_lat, wrong_soft

                # Tokenize teacher summary → target token embeddings
                enc = base_tokenizer(target_texts[bucket_id], return_tensors="pt")
                target_ids = enc["input_ids"].squeeze(0).long().to(device)
                del enc
                target_embeds = embedding_layer(target_ids)                     # (T, D)

                # Concatenate: [framed_prefix | target_token_embeds]
                prefix_len = framed.shape[0]   # K+2
                inputs_embeds = torch.cat([framed, target_embeds], dim=0)
                inputs_embeds_wrong = torch.cat([framed_wrong, target_embeds], dim=0)
                del framed, framed_wrong, target_embeds

                item = {
                    "inputs_embeds": inputs_embeds,
                    "inputs_embeds_wrong": inputs_embeds_wrong,
                    "target_ids": target_ids,
                    "prefix_len": prefix_len,
                }

                losses = trainer.train_step(item)
                epoch_task_loss += losses["task"]
                epoch_sep_loss += losses["sep"]
                count += 1
                print(f"    L_task={losses['task']:.4f}  L_sep={losses['sep']:.4f}")

                # Task 2b: purge bucket tensors and MPS pool after every step
                del item, inputs_embeds, inputs_embeds_wrong, target_ids, losses
                if device == "mps":
                    torch.mps.empty_cache()

            except Exception as exc:
                print(f"  Skipped {bucket_id}: {exc}")

        if count:
            print(f"  Epoch {epoch+1} avg — L_task={epoch_task_loss/count:.4f}  "
                  f"L_sep={epoch_sep_loss/count:.4f}")

    # ── Save LoRA weights ──────────────────────────────────────────────────────
    lora_params = {n: p for n, p in base_model.named_parameters() if "lora" in n.lower()}
    torch.save(lora_params, output)
    print(f"[train] LoRA weights saved to {output}  ({len(lora_params)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--output", default=None, type=Path,
                        help="Output path for LoRA weights (default: <repo>/.libucks/base_receiver_lora.pt)")
    args = parser.parse_args()

    output = args.output or (Path(args.repo) / ".libucks" / "base_receiver_lora.pt")
    asyncio.run(main(Path(args.repo), args.epochs, args.device, output))
