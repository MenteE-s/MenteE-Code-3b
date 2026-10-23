# MenteE Code 3B — Training Pipeline

End-to-end pipeline: download free trajectory datasets → preprocess → train → benchmark → upload.

## Phase 1: Local smoke test (this machine)

Verify the entire pipeline works before spending money on vast.ai.

```powershell
# 1. download a small slice (~200 rows per source)
python scripts/download_datasets.py --out data/external --max-rows 200

# 2. convert to mente format
python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl

# 3. smoke-test training (tiny LoRA, ~5 min on CPU, faster on GPU)
python scripts/train_smoke.py --data data/external/prepared.jsonl --max-samples 50

# 4. verify the model loads and generates
python scripts/train_smoke.py --data datasets/collectors/syab/train.jsonl --max-samples 10
```

If all 4 steps run without errors, the pipeline is ready for vast.ai.

## Phase 2: Full pipeline (vast.ai)

```bash
# 1. download full datasets (no --max-rows)
python scripts/download_datasets.py --out data/external

# 2. preprocess everything
python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl

# 3. merge with your own generated trajectories
python -m src.collect merge --inputs data/external/prepared.jsonl datasets/collectors/syab/train.jsonl --out data/training/train.jsonl

# 4. train (full LoRA on Qwen2.5-Coder-3B or Qwen3-4B)
#    use your vast.ai training script here
#    recommended: QLoRA, rank 64, 3 epochs, lr 2e-4

# 5. benchmark on SWE-bench Verified (subset of 50)
python scripts/benchmark.py --model outputs/full_run --tasks 50

# 6. upload to HuggingFace (private)
python scripts/upload_model.py --model-dir outputs/full_run --repo MenteE-s/MenteE-Code-3B
```

## Where to save the model

**HuggingFace Hub (private)** — recommended. Free, versioned, easy to download.
- Create a private repo: `MenteE-s/MenteE-Code-3B`
- Upload with `scripts/upload_model.py`
- Download on any machine with: `huggingface-cli download MenteE-s/MenteE-Code-3B`

## Where to run inference (cheap/free)

| Option | Cost | Best for |
|--------|------|----------|
| **HuggingFace Inference Endpoints** | ~$0.50-1/hr (small GPU) | Dedicated, low-latency testing |
| **RunPod Serverless** | ~$0.0002/sec | Pay-per-use, good for benchmarking |
| **Together AI** | ~$0.10/M tokens | Hosted, no infra needed |
| **Modal** | ~$0.0003/sec | Serverless, Python-native |
| **Ollama (local)** | Free | Quick local testing on your machine |
| **Vast.ai (keep instance)** | ~$0.30-0.50/hr | If you already have the instance |

**Recommendation for testing:** HuggingFace Inference Endpoints or Ollama (local).
**Recommendation for production:** Together AI or RunPod serverless (cheapest per-token).

## Vast.ai setup checklist

1. Rent an RTX 4090 (24GB) — ~$0.40/hr
2. `git clone` your repo
3. `python -m venv .venv && pip install -r requirements-train.txt`
4. Run the full pipeline (steps 1-4 above)
5. Train: ~2-4 hours for 50K samples with LoRA
6. Upload to HuggingFace
7. Terminate the instance (stop paying)

## Cost estimate (full run)

| Step | Cost |
|------|------|
| Vast.ai GPU (4 hours) | ~$1.60 |
| HuggingFace storage | Free (private repo) |
| Inference testing | ~$0.50 (Together AI, ~5M tokens) |
| **Total** | **~$2.10** |

With the $10 OpenRouter budget for trajectory generation + ~$2 for training/inference, you can go from zero to a trained, benchmarked, hosted 3B coding model for under $15 total.