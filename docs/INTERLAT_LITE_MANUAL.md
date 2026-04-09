# Interlat-Lite Operational Runbook

## 0. What This Is

Interlat-Lite replaces the broken `decode()` injection path with a trained Base-model receiver.
Two standalone scripts bridge the gap until Phase 13 wires this into the CLI:

| Script | Purpose |
|---|---|
| `scripts/train_lora_receiver.py` | Fine-tunes Base receiver with LoRA on your repo's buckets |
| `scripts/query_interlat.py` | Runs a query through the full Interlat-Lite pipeline |

> **CLI not yet wired.** `libucks train-adapter` and `libucks query` still use the old InfoNCE/`decode()` path. Use the scripts below.

---

## 1. Prerequisites

### Hardware
- Apple M1 or better (≥8 GB unified memory), OR CUDA GPU with ≥4 GB VRAM
- Both 0.5B models run natively without quantization; peak usage ~3–4 GB

### Models (download before running)
```
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B
```

### Python environment
```
pip install "libucks[latent]"   # installs peft, bitsandbytes, transformers, torch
```

### Required artifacts (must exist before training)
| Artifact | Created by |
|---|---|
| `<repo>/.libucks/adapter.pt` | `libucks train-adapter --repo <repo>` |
| `<repo>/.libucks/*.bucket` files | `libucks init --repo <repo>` |

### Environment variable
```
export ANTHROPIC_API_KEY=sk-...   # used by TextStrategy to generate curriculum prose
```

---

## 2. Strict Order of Operations

Run these steps in order. Do not skip steps.

### Step 1 — Initialise the repo (if not already done)
```bash
libucks init --repo /path/to/your/repo
```
Verify: `.libucks/registry.json` exists and contains at least one bucket.

### Step 2 — Train the CommunicationAdapter (if not already done)
```bash
libucks train-adapter --repo /path/to/your/repo
```
Verify: `.libucks/adapter.pt` exists.

### Step 3 — Fine-tune the LoRA receiver
```bash
python scripts/train_lora_receiver.py \
  --repo /path/to/your/repo \
  --epochs 3 \
  --device mps
```
Expected console output each bucket: `L_task=X.XXXX  L_sep=X.XXXX`
Verify: `.libucks/base_receiver_lora.pt` is written at the end.

### Step 4 — Query via Interlat-Lite
```bash
python scripts/query_interlat.py \
  --repo /path/to/your/repo \
  --query "What does the authentication module do?" \
  --device mps
```
Answer is printed to stdout. All diagnostic logs go to stderr.

---

## 3. Testing Methodology

### 3.1 Sanity check — shape and type
Before running on a real repo, verify the pipeline doesn't crash on a trivial query:
```bash
python scripts/query_interlat.py \
  --repo /path/to/your/repo \
  --query "hello" \
  --device cpu \
  --lora /dev/null
```
A coherent (possibly weak) English response — not BPE fragments — confirms the Base model path is working.

### 3.2 Character-soup test — compare old vs new
Run the same query through the old `libucks query` and the new script side-by-side:
```bash
libucks query --repo /path/to/your/repo "What does X do?"       # old path
python scripts/query_interlat.py --repo /path/to/your/repo --query "What does X do?"  # new path
```
The new path should return readable sentences. If you still see BPE fragments (`\.2m $s q)\`), the LoRA weights were not loaded — check that `base_receiver_lora.pt` exists.

### 3.3 Unit tests (no GPU required)
```bash
pytest tests/unit/test_curriculum.py \
       tests/unit/test_interlat_adapter.py \
       tests/unit/test_lsep_loss.py \
       tests/unit/test_lora_trainer.py \
       tests/unit/test_latent_strategy.py \
  -v
```
All tests should pass on CPU. The integration test in `test_latent_strategy.py` is marked `@pytest.mark.skip` and requires a GPU.

### 3.4 Loss convergence check
During `train_lora_receiver.py`, L_task should decrease across epochs. If it does not decrease after epoch 1:
- Confirm `adapter.pt` was trained (not zero-initialised)
- Try `--epochs 5`
- Reduce `lr` to `1e-5` by editing the script

---

## 4. Artifacts Produced

| File | Contents |
|---|---|
| `.libucks/base_receiver_lora.pt` | LoRA weight tensors for `q_proj` and `v_proj` in the Base receiver |

The LoRA weights are automatically loaded by `query_interlat.py` from the default path. Pass `--lora <path>` to override.

---

## 5. What Phase 13 Will Wire

Phase 13 will make the glue scripts unnecessary by:
- Adding `base_model` field to `ModelConfig`
- Adding `libucks train-receiver` command
- Updating `Translator._synthesize_latent()` to call `strategy.receive()` instead of `strategy.decode()`
- Auto-loading LoRA weights in `_run_query()`
