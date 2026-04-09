import asyncio
import argparse
import sys
from pathlib import Path


async def main(repo_path: Path, query_text: str, lora_path: Path, top_k: int, device: str):
    import torch
    from libucks.config import Config
    from libucks.thinking.model_manager import ModelManager
    from libucks.thinking.latent_strategy import LatentStrategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.central_agent import CentralAgent
    from libucks.librarian import Librarian
    from libucks.query_orchestrator import QueryOrchestrator

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"

    # ── Load both models ──────────────────────────────────────────────────────
    print(f"[query] Loading Instruct encoder ({device})...", file=sys.stderr)
    mgr = ModelManager()
    mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="4bit", device=device)

    print(f"[query] Loading Base receiver ({device})...", file=sys.stderr)
    mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="4bit", device=device)

    # ── Restore LoRA weights ──────────────────────────────────────────────────
    # _inject_lora must be called first so the model has lora_A/lora_B parameters
    # before we try to load into them — otherwise named_parameters() has no LoRA
    # keys and 0 tensors are restored.
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer, _inject_lora
    base_model = mgr.get_base_model()
    _inject_lora(base_model, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), r=4, alpha=4)
    # Freeze non-LoRA params to match training configuration
    for name, param in base_model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)

    if lora_path.exists():
        print(f"[query] Loading LoRA weights from {lora_path}...", file=sys.stderr)
        lora_state = torch.load(lora_path, map_location=device, weights_only=True)
        current = dict(base_model.named_parameters())
        loaded = 0
        for name, tensor in lora_state.items():
            if name in current:
                current[name].data.copy_(tensor)
                loaded += 1
        print(f"[query] Restored {loaded} LoRA tensors", file=sys.stderr)
    else:
        print(f"[query] WARNING: {lora_path} not found — using untrained receiver", file=sys.stderr)

    strategy = LatentStrategy(mgr)

    # Token recycling: native Qwen chat-boundary tokens as frame markers.
    base_tokenizer = mgr.get_base_tokenizer()
    base_model = mgr.get_base_model()
    bop_id = base_tokenizer.convert_tokens_to_ids("<|im_start|>")
    eop_id = base_tokenizer.convert_tokens_to_ids("<|im_end|>")
    embedding_layer = base_model.model.embed_tokens
    bop_embed = embedding_layer(torch.tensor([bop_id], device=device)).squeeze(0).detach()
    eop_embed = embedding_layer(torch.tensor([eop_id], device=device)).squeeze(0).detach()

    # ── Load buckets + routing ────────────────────────────────────────────────
    registry = BucketRegistry(repo_path / cfg.paths.registry_file)
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        print("[query] ERROR: No buckets. Run `libucks init` first.", file=sys.stderr)
        return

    _real_stdout = sys.stdout; sys.stdout = sys.stderr
    try:
        embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    finally:
        sys.stdout = _real_stdout

    # ── Load adapter ──────────────────────────────────────────────────────────
    hidden_dim = mgr.get_base_model().config.hidden_size
    adapter = CommunicationAdapter(hidden_dim=hidden_dim)
    adapter.load_saved_weights(bucket_dir / "adapter.pt")
    adapter_dtype = torch.float16 if device == "mps" else None
    adapter = adapter.to(device=device, dtype=adapter_dtype)

    # ── Build librarians + orchestrator ───────────────────────────────────────
    agent = CentralAgent(registry, cfg, embed_fn=embedder.embed)
    librarians = {}
    for bid in bucket_ids:
        lib = Librarian(
            bucket_id=bid, store=store, registry=registry,
            strategy=strategy, embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
        )
        librarians[bid] = lib
        agent.register_librarian(bid, lib)

    orchestrator = QueryOrchestrator(
        central_agent=agent, librarians=librarians,
        embed_fn=embedder.embed, top_k=top_k,
    )

    # ── Query → representations → frame → receive ─────────────────────────────
    print(f'[query] Routing: "{query_text}"', file=sys.stderr)
    representations = await orchestrator.query(query_text)
    print(f"[query] {len(representations)} representations from Librarians", file=sys.stderr)

    if not representations:
        print("No relevant context found.", file=sys.stderr)
        return

    # Merge via adapter, frame with <bop>/<eop>, decode via Base receiver
    contiguous_reps = [r.contiguous() for r in representations]
    with torch.no_grad():
        soft_prompt = adapter(contiguous_reps)            # (K, D)
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)   # (K+2, D)

    print("[query] Calling strategy.receive()...", file=sys.stderr)
    answer = await strategy.receive(framed)

    # Answer to stdout; all logs to stderr
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--lora", default=None, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    lora = args.lora or (Path(args.repo) / ".libucks" / "base_receiver_lora.pt")
    asyncio.run(main(Path(args.repo), args.query, lora, args.top_k, args.device))