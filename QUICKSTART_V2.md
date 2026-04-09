# libucks V2 Quickstart — Latent Space Engine

This runbook walks you through transitioning a V1 (text-based) libucks installation to the V2 Latent Space architecture and starting the live engine.

**What changes in V2.** Librarians no longer return natural-language strings. They return raw `torch.Tensor` hidden states from a local Qwen2.5-3B model. These tensors are merged by the `CommunicationAdapter` and compressed by the `LatentCompressor` before the Translator decodes them into a final answer. The Anthropic API is still used as the V1 teacher during adapter training, but is no longer required for live inference.

---

## 1. Prerequisites

### Python and hardware

- Python 3.11+
- Apple Silicon (M1/M2/M3) with at least **8 GB unified memory** recommended — the 3B model loads in **float16** by default (~6 GB), staying well under Metal's 4 GB per-allocation limit
- CUDA GPU (≥ 8 GB VRAM) also supported; CPU fallback works but is slow

> **float16 on MPS is automatic.** `ModelManager` detects MPS and loads the model in `torch.float16` regardless of the `quantization` config value. You do not need to set anything. The `CommunicationAdapter` is also cast to float16 at startup to match. Do not override this manually.

### Install the latent dependencies

V2 model inference depends on PyTorch and Transformers, which are declared as optional extras to keep V1 installs lean.

```bash
pip install -e ".[latent]"
```

This installs:
- `torch >= 2.2`
- `transformers >= 4.40`
- `accelerate >= 0.28`

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected on Apple Silicon:
```
2.x.x  True
```

### Anthropic API key

The `--creative` training step uses the V1 TextStrategy as a teacher to generate multi-perspective training triplets. Ensure `ANTHROPIC_API_KEY` is set in your environment:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

This is only required during `libucks train-adapter`. The live inference engine does not call the Anthropic API.

---

## 2. Configuration

Open or create `.libucks/config.toml` at the root of your repository. Add or update the `[model]` section as follows:

```toml
[model]
# V1 teacher — used during train-adapter --creative only
anthropic_model = "claude-haiku-4-5-20251001"

# Sentence-transformers model for bucket routing (unchanged from V1)
embedding_model = "all-MiniLM-L6-v2"

# V2: local Qwen2.5-3B model for hidden-state reasoning
local_model    = "Qwen/Qwen2.5-3B-Instruct"
strategy       = "latent"       # activates V2 — set to "text" to revert to V1
device         = "mps"          # "mps" for Apple Silicon | "cuda" | "cpu"
quantization   = "none"         # "none" | "4bit" | "8bit"  (4bit saves ~3 GB VRAM)

# Latent compressor: squashes (L, 2048) → (8, 2048) before the adapter
compression_steps = 8           # set to 0 to disable compression
```

**Device reference:**

| Hardware | `device` value |
|---|---|
| Apple Silicon (M1/M2/M3) | `"mps"` |
| NVIDIA GPU | `"cuda"` |
| CPU only | `"cpu"` |

> **To revert to V1 at any time**, set `strategy = "text"`. No other changes are needed — all V1 code paths remain intact.

---

## 3. Seed the buckets (if starting fresh)

If you have not yet run `libucks init`, index your repository now. This is identical to V1:

```bash
libucks init --local /path/to/your/repo
```

This populates `.libucks/` with bucket prose and a centroid registry. Training the adapter requires at least a few buckets to exist.

---

## 4. Train the Adapter

> **This step is required before you can query.** Without a trained adapter, the `CommunicationAdapter` starts from random weights. The Translator will receive a meaningless projection of Librarian latents and produce incoherent or hallucinated output. Run `train-adapter --creative` at least once after `libucks init` and again any time you re-seed buckets.

The `--creative` flag activates contrastive training: for each bucket, the V1 teacher generates three complementary perspectives (Summary, Logic Flow, Dependency Map), and topically-adjacent buckets are mined as hard negatives for InfoNCE loss.

```bash
libucks train-adapter --creative --epochs 3
```

> **First run warning:** HuggingFace will download `Qwen/Qwen2.5-3B-Instruct` (~6 GB) to your local cache (`~/.cache/huggingface/`). This is a one-time download. Subsequent runs load from cache and start within seconds.

**What happens during training:**

1. Each bucket's prose is sent to the Anthropic API (the V1 teacher) once per perspective to generate ground-truth text.
2. Those texts are encoded by the local Qwen model into hidden-state tensors.
3. Topically adjacent buckets (cosine similarity in `[0.25, 0.65]`) are encoded as hard negatives.
4. The `CommunicationAdapter` is trained via InfoNCE loss to pull its output toward the correct positive latent and away from the negatives.
5. Trained weights are saved to `.libucks/adapter.pt`.

**Faster run for testing** (1 epoch, smaller learning rate):
```bash
libucks train-adapter --creative --epochs 1
```

**Training output example:**
```
[libucks] Creative contrastive training on 12 buckets for 3 epoch(s)...
  Generating sample 1/12: bucket-auth
  Generating sample 2/12: bucket-db
  ...
Training complete. Loss: 0.6931 → 0.1204. Saved to .libucks/adapter.pt
```

A loss reduction from ~0.69 (random) toward ~0.10 indicates the adapter has learned to distinguish correct from incorrect latent directions.

---

## 5. Boot the Engine

Start the libucks MCP server over stdio. This is the same command as V1; the server automatically loads the local model and trained adapter when `strategy = "latent"` is set in config.

```bash
libucks serve
```

**Startup output (stderr):**
```
[libucks] repo=/path/to/repo  registry=.libucks/registry.json  buckets=.libucks
[libucks] startup recovery complete, HEAD=a1b2c3d4
```

If `.libucks/adapter.pt` exists, it is loaded automatically. If not, the adapter starts from random weights (reduced quality).

### Running as a background daemon

```bash
nohup libucks serve > /dev/null 2>> .libucks/libucks.log &
echo $! > .libucks/server.pid
```

To stop:
```bash
kill $(cat .libucks/server.pid)
```

### Claude Desktop integration

Add the following to your `claude_desktop_config.json` to register libucks as an MCP server. The `LIBUCKS_REPO_PATH` variable tells the server which repository to load:

```json
{
  "mcpServers": {
    "libucks": {
      "command": "libucks",
      "args": ["serve"],
      "env": {
        "LIBUCKS_REPO_PATH": "/absolute/path/to/your/repo",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Restart Claude Desktop after editing this file.

---

## 6. Direct CLI Query (Recommended for Apple Silicon)

> **Why use this instead of the MCP Inspector UI?**
> The MCP Inspector web UI enforces a hard 60-second timeout on every tool call. On Apple Silicon, serialized Qwen 3B inference across 3 Librarians can take 60–90 seconds — consistently hitting that wall. The `libucks query` command runs the identical pipeline as a plain terminal process with **no timeout**, so the model can take as long as it needs. Once you have real latency numbers from a successful terminal run, you can tune `max_length` and `max_new_tokens` in `latent_strategy.py` to fit the UI limit.

### Syntax

```bash
libucks query "your question here"
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--repo PATH` | git root of `cwd` | Path to the repository to query. Defaults to the git root containing your current directory. |
| `--top-k N` | `3` | Number of memory buckets to consult. Use `--top-k 1` for the fastest possible single-bucket test. |

### Examples

```bash
# Basic query against the current repository
libucks query "How does the dependency injection system work?"

# Query a specific repo, consult only the single most relevant bucket
libucks query "How does authentication work?" --repo ~/projects/myapp --top-k 1

# Save the answer to a file
libucks query "Explain the database migration strategy" > answer.txt
```

### stdout / stderr split

Progress logs and inference checkpoints go to **stderr**; the final answer goes to **stdout**. This means you can pipe or redirect the answer cleanly without capturing noise:

```bash
# Terminal shows inference logs live; only the answer lands in answer.txt
libucks query "How does the caching layer work?" > answer.txt

# Suppress all logs, capture only the answer
libucks query "Explain the test strategy" 2>/dev/null > answer.txt
```

When running without redirection, the full trace is visible in real time:

```
[libucks] repo=/path/to/repo  strategy=latent
[libucks] 12 buckets loaded
[libucks] strategy ready
[libucks] routing: "How does the dependency injection system work?"
[libucks:latent] reason: tokenizing (1842 chars, device=mps)
[libucks:latent] reason: forward pass (seq_len=256)
[libucks:latent] reason: forward pass complete, hidden=(256, 2048)
[libucks:latent] reason: compressing (256 → 8 steps)
...
[libucks:latent] decode: generate complete, output_ids=(1, 160)
[libucks] 3 representations, synthesizing...

The dependency injection system uses a constructor-injection pattern...
```

---

## 7. MCP Live Test

Once the server is running (or registered in Claude Desktop), you can verify the full V2 pipeline with two built-in MCP tools.

### libucks_status — health check

Confirm the server sees your buckets:

```
Tool: libucks_status
Arguments: {}
```

Expected response:
```json
{
  "bucket_count": 12,
  "total_tokens": 48302,
  "buckets": {
    "bucket-auth": { "token_count": 4210 },
    "bucket-db":   { "token_count": 3890 },
    ...
  }
}
```

### libucks_query — end-to-end latent query

Send a natural-language question. The request travels through the full V2 pipeline:
`embed query → route to top-K buckets → LatentStrategy.reason() → compress → CommunicationAdapter → LatentStrategy.decode()`.

```
Tool: libucks_query
Arguments: { "query": "How does the authentication module validate JWT tokens?", "top_k": 3 }
```

A coherent, context-grounded answer confirms the pipeline is functioning end-to-end.

### Checking the data flow

If you want to trace what the engine is doing, tail the log:

```bash
tail -f .libucks/libucks.log
```

Or watch stderr if running in the foreground — all Librarian reasoning happens silently in latent space; only the final Translator output reaches the MCP client.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | Optional deps not installed | `pip install -e ".[latent]"` |
| `RuntimeError: MPS not available` | Running on Intel Mac or Linux | Set `device = "cuda"` or `device = "cpu"` |
| Incoherent or hallucinated answers | Adapter not trained — random weights produce garbage | Run `libucks train-adapter --creative --epochs 3` |
| Poor answer quality after training | Too few epochs or too few buckets | Run `libucks train-adapter --creative --epochs 5` |
| Slow first query (~30–60s) | Model loading + initial KV cache warm-up | Normal — subsequent queries are faster |
| MCP Inspector "Request timed out" | 60s UI timeout exceeded by serial Qwen inference | Use `libucks query "..."` from the terminal instead — no timeout |
| `.libucks/adapter.pt` not found | Training hasn't been run yet | Run `libucks train-adapter --creative` |
| OOM on MPS | Unified memory pressure from other processes | Set `quantization = "4bit"` or reduce `compression_steps = 4` |
