# MenteE Code 3B — Full Pipeline (Vast.ai → HuggingFace)

## HuggingFace repo names (create these before starting)

| What | Repo name | Type |
|------|-----------|------|
| **Model** | `MenteEAI/MenteE-Code-3B` | Model (private) |
| **Dataset** | `MenteEAI/MenteE-Code-3B-Dataset` | Dataset (private) |
| **GGUF (for Ollama)** | `MenteEAI/MenteE-Code-3B-GGUF` | Model (private) |

Create them at: https://huggingface.co/new (select "Model" or "Dataset", check "Private")

---

## Phase 1: Vast.ai setup (one-time, ~15 min)

### 1.1 Rent a GPU
```
vast.ai → search "RTX 4090 / RTX 5090" → sort by price → rent
Template: PyTorch (Vast) — NOT raw CUDA (avoids huge pip installs)
SSH into the instance
```

> **Speed check (learned the hard way):** some cheap hosts are shared/throttled.
> After training starts, look at the `s/it` in the progress bar:
> - ~2–4 s/it → good host, let it run
> - **>8 s/it → shared GPU. Consider destroying and picking another host.**
> Every minute spent deciding early saves hours later.

### 1.2 Install dependencies
```bash
git clone https://github.com/MenteE-s/MenteE-Code-3b.git
cd MenteE-Code-3b
pip install -r requirements-train.txt
hf auth login   # NOT huggingface-cli (not installed on vast templates)
```

### 1.3 Verify GPU
```bash
nvidia-smi  # should show the rented GPU + VRAM
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Phase 2: Download + preprocess data (~45 min)

### 2.1 Download free trajectory datasets
```bash
python scripts/download_datasets.py --out data/external --max-rows 25000
```
Downloads 25,000 rows from each of 4 sources = ~100,000 raw trajectories.
No `--max-rows` would download TBs — don't do that.

### 2.2 Convert to mente format (~10–15 min, CPU-bound — the GPU is idle here, that's normal)
```bash
python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl
# expect: ~78,000 converted
```

### 2.3 Identity data + merge
```bash
python scripts/generate_identity_data.py --out data/identity/identity.jsonl --count 500

# NOTE: the merge writer refuses to overwrite (append-only by design).
# On a fresh instance this file won't exist; on a reused one, delete first:
rm -f data/training/train.jsonl

python -m src.collect merge --inputs data/external/prepared.jsonl datasets/collectors/syab/train.jsonl data/identity/identity.jsonl --out data/training/train.jsonl
# expect: ~53,500 accepted (25K dup-content dropped by dedup, 0 invalid)
```

### 2.4 Shuffle in place (CRITICAL — do not skip)
```bash
shuf -o data/training/train.jsonl data/training/train.jsonl
wc -l data/training/train.jsonl   # still 53567
```
Why: `--max-samples N` takes the FIRST N rows. Unshuffled, that's 100% nvidia_zero
and zero identity/syab rows. `shuf -o` edits the file in place so it needs no extra
disk space (plain `shuf a > b` can fail with "No space left on device" on a 30 GB disk).

---

## Phase 3: Train (~5 hours)

### 3.1 Run training
```bash
python scripts/train_smoke.py \
  --data data/training/train.jsonl \
  --max-samples 20000 \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output outputs/full_run \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 4 \
  --max-seq-length 4096 \
  --context-length 131072
```

- 20,000 shuffled samples × 1 epoch ≈ **1,250 steps**. Loss should hit ~0.4 within
  the first 100 steps; a 3-epoch full run adds <5% quality for 5x the time.
- `--context-length 131072` writes YaRN rope-scaling into config.json AFTER training
  so the published model advertises 128K context.
- The `warmup_ratio (skipped)` line is harmless (transformers 5.x compat).
- The `66785 > 32768` tokenizer warning is harmless (rows are truncated to 4096).

### 3.2 Monitor training
```bash
# In another terminal
watch -n 10 nvidia-smi                                  # GPU util should be 90%+
df -h /                                                 # keep >2 GB free
```

Healthy signs: `loss` falling (1.6 → 0.6 in ~30 steps), `grad_norm` < 1, no NaN.

### 3.3 Verify model saved
```bash
ls outputs/full_run/      # config.json, model safetensors, tokenizer, adapter
du -sh outputs/full_run/  # ~6 GB
grep -A3 rope_scaling outputs/full_run/config.json   # YaRN block should be present
```

---

## Phase 4: Upload to HuggingFace (~10 min)

### 4.1 Upload the model
```bash
python scripts/upload_model.py --model-dir outputs/full_run --repo MenteEAI/MenteE-Code-3B
```

### 4.3 Benchmark on SWE-bench Verified
```bash
python scripts/benchmark.py --model MenteEAI/MenteE-Code-3B --tasks 50
```

Runs 50 SWE-bench Verified tasks and reports resolve rate vs baselines (GPT-4o, Claude, etc.).

### 4.3 Verify on HuggingFace
```
https://huggingface.co/MenteEAI/MenteE-Code-3B
```

---

### 4.5 Export GGUF — the "lightweight" release (~2 GB, runs anywhere)

Do this on your LOCAL machine (the vast instance disk is tight, and quantizing
needs no GPU):

```bash
# 1. pull the trained model you just uploaded
hf download MenteEAI/MenteE-Code-3B --local-dir model-full   # or: huggingface-cli download

# 2. convert with llama.cpp
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py model-full --outfile mentee-f16.gguf --outtype f16

# 3. quantize (Q4_K_M is the sweet spot: ~2 GB, minimal quality loss)
./llama.cpp/llama-quantize mentee-f16.gguf MenteE-Code-3B-Q8_0.gguf Q8_0      # ~3.3 GB, near-lossless
./llama.cpp/llama-quantize mentee-f16.gguf MenteE-Code-3B-Q4_K_M.gguf Q4_K_M  # ~2.0 GB, recommended

# 4. upload both to the GGUF repo
hf upload MenteEAI/MenteE-Code-3B-GGUF MenteE-Code-3B-Q4_K_M.gguf
hf upload MenteEAI/MenteE-Code-3B-GGUF MenteE-Code-3B-Q8_0.gguf
```

### Ollama Modelfile (bakes in the MenteE identity — users get it with one command)

```
# File: Modelfile
FROM ./MenteE-Code-3B-Q4_K_M.gguf
SYSTEM """You are MenteE Code, a coding assistant built by MenteE. You always identify as MenteE Code, created by MenteE."""
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
```

```bash
ollama create mentee-code -f Modelfile
ollama run mentee-code
```

> Serving tip: 32K context is the fast/cheap default. 128K (YaRN) works but costs
> ~4x KV-cache memory — expose it as an option, not the default.

---

## Phase 5: Terminate Vast.ai (stop paying)

```bash
# On vast.ai dashboard: terminate instance
# Or via CLI:
vastai destroy instance <instance_id>
```

Safe to destroy any time BEFORE Phase 4 — everything regenerates:
scripts live in git, data re-downloads from HuggingFace, the model re-downloads
on next launch. Nothing irreplaceable ever lives on the instance.

---

## Gotchas (all learned in real runs — read before debugging)

| Symptom | Cause | Fix |
|---|---|---|
| `AppendError: train.jsonl already contains data` | merge writer is append-only by design | `rm -f data/training/train.jsonl` before merging |
| `KeyError: 'rope_type'` on model load | transformers 5.x rejects `rope_scaling` kwarg at load | fixed — YaRN is applied to config.json post-training |
| `SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` | trl 0.12+ renamed it | fixed — script uses `processing_class` |
| `SFTConfig ... unexpected keyword 'warmup_ratio'` | transformers 5.x dropped fields | harmless — script filters kwargs by signature and prints what it skipped |
| `protobuf` version conflict warnings | vLLM vs unsloth-zoo metadata | cosmetic, ignore. Do NOT downgrade protobuf (breaks vllm) |
| unsloth `ConstantLengthDataset` import error | unsloth needs older trl | don't fight it — `train_smoke.py` produces the same model, ~2x slower |
| `shuf: No space left on device` | 30 GB disk full | clean caches (below), then `shuf -o file file` (in place) |
| 16 s/it instead of ~3 s/it | shared/throttled GPU host | destroy, rent another host, re-test early |
| `huggingface-cli: command not found` | vast template ships only new CLI | use `hf auth login` |

### Disk cleanup on a 30 GB instance (safe deletions)
```bash
rm -rf /root/.cache/huggingface/datasets   # tokenized cache, regenerated automatically
rm -rf /root/.cache/pip
rm -rf data/external/nvidia_zero data/external/openhands data/external/r2e_gym data/external/swe_agent  # raw sources; prepared.jsonl is what we use
# NEVER delete /root/.cache/huggingface/hub — that's the base model
```

---

## Phase 6: Host for free inference (Modal)

### 6.1 Deploy
```bash
pip install modal
modal token new
modal deploy scripts/modal_deploy.py
```

### 6.2 Download model to Modal volume
```bash
modal run scripts/modal_deploy.py::download_model
```

### 6.3 Get endpoint URL
```
Modal dashboard → mentee-code-3b → copy endpoint URL
Example: https://your-username--mentee-code-3b-serve.modal.run
```

### 6.4 Test with OpenCode
```json
{
  "base_url": "https://your-username--mentee-code-3b-serve.modal.run/v1",
  "model": "MenteE-Code-3B",
  "api_key": "not-needed"
}
```

---

## Phase 7: Collect user data (flywheel)

### 7.1 Run logging proxy
```bash
python scripts/log_proxy.py --backend https://your-modal-endpoint/v1 --port 8080 --push-hf MenteEAI/MenteE-Code-3B-Dataset
```

### 7.2 Users connect via proxy
```
OpenCode base_url: http://your-server:8080/v1
```

### 7.3 Convert conversations to training data
```bash
python scripts/convert_conversations.py --in data/collected/conversations.jsonl --out data/collected/prepared.jsonl --append-to datasets/collectors/community/train.jsonl --collector community
```

### 7.4 Retrain (repeat Phase 2-5)
```bash
python -m src.collect merge --inputs data/collected/prepared.jsonl data/training/train.jsonl --out data/training/train_v2.jsonl
# upload to vast.ai → train → upload new model → repeat
```

---

## Full one-line commands

### Smoke test (local, ~5 min)
```powershell
python scripts/download_datasets.py --out data/external --max-rows 200 && python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl && python scripts/generate_identity_data.py --out data/identity/identity.jsonl --count 500 && python -m src.collect merge --inputs data/external/prepared.jsonl datasets/collectors/syab/train.jsonl data/identity/identity.jsonl --out data/training/train.jsonl && python scripts/train_smoke.py --data data/training/train.jsonl --max-samples 50
```

### Production (vast.ai, ~4 hours, ~$2)
```bash
python scripts/download_datasets.py --out data/external --max-rows 25000 && python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl && python scripts/generate_identity_data.py --out data/identity/identity.jsonl --count 500 && python -m src.collect merge --inputs data/external/prepared.jsonl datasets/collectors/syab/train.jsonl data/identity/identity.jsonl --out data/training/train.jsonl && python scripts/train_smoke.py --data data/training/train.jsonl --max-samples 999999 --model Qwen/Qwen2.5-Coder-3B-Instruct --output outputs/full_run
```

### Upload
```bash
python scripts/upload_model.py --model-dir outputs/full_run --repo MenteEAI/MenteE-Code-3B
```

---

## Cost summary

| Step | Cost |
|------|------|
| Vast.ai GPU (4 hours) | ~$1.60 |
| HuggingFace storage | Free |
| Modal inference (free tier) | $0 |
| OpenRouter $10 (optional, for generation) | $10.00 |
| **Total (no generation)** | **~$1.60** |
| **Total (with generation)** | **~$11.60** |

## Full flow (one line)

```
Vast.ai (train) → HuggingFace (store) → Modal (serve) → OpenCode (use) → log_proxy (collect) → convert (preprocess) → Vast.ai (retrain) → ∞
```