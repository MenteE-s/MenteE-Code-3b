# MenteE Code 3B — Full Pipeline (Vast.ai → HuggingFace)

## HuggingFace repo names (create these before starting)

| What | Repo name | Type |
|------|-----------|------|
| **Model** | `MenteEAI/MenteE-Code-3B` | Model (private) |
| **Dataset** | `MenteEAI/MenteE-Code-3B-Dataset` | Dataset (private) |
| **GGUF (for Ollama)** | `MenteEAI/MenteE-Code-3B-GGUF` | Model (private) |

Create them at: https://huggingface.co/new (select "Model" or "Dataset", check "Private")

---

## Phase 1: Vast.ai setup (one-time, ~10 min)

### 1.1 Rent a GPU
```
vast.ai → search "RTX 4090" → sort by price → rent cheapest (~$0.30-0.40/hr)
SSH into the instance
```

### 1.2 Install dependencies
```bash
git clone https://github.com/MenteE-s/MenteE-Code-3b.git
cd MenteE-Code-3b
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
huggingface-cli login  # paste your HF token
```

### 1.3 Verify GPU
```bash
nvidia-smi  # should show RTX 4090, 24GB
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Phase 2: Download + preprocess data (~30 min)

### 2.1 Download free trajectory datasets (100K rows)
```bash
python scripts/download_datasets.py --out data/external --max-rows 25000
```
Downloads 25,000 rows from each of 4 sources = ~100,000 trajectories total.
No `--max-rows` would download TBs — don't do that.

### 2.2 Convert to mente format
```bash
python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl
```

### 2.3 Merge with your generated trajectories
```bash
python -m src.collect merge --inputs data/external/prepared.jsonl datasets/collectors/syab/train.jsonl --out data/training/train.jsonl
```

### 2.4 Verify the dataset
```bash
wc -l data/training/train.jsonl  # should be ~100,032
python -m src.collect verify --path data/training/train.jsonl
```

---

## Phase 3: Train (~4 hours, ~$1.60)

### 3.1 Run training
```bash
python scripts/train_smoke.py \
  --data data/training/train.jsonl \
  --max-samples 999999 \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output outputs/full_run
```

**Note:** Modify `train_smoke.py` for full run:
- Set `num_train_epochs=3`
- Set `per_device_train_batch_size=4`
- Set `gradient_accumulation_steps=4`
- Set `learning_rate=2e-4`
- Set `max_seq_length=4096`
- Set `bf16=True`

### 3.2 Monitor training
```bash
# In another terminal
watch -n 10 nvidia-smi  # GPU utilization
tail -f outputs/full_run/trainer_state.json  # training progress
```

### 3.3 Verify model saved
```bash
ls outputs/full_run/  # should have model weights, tokenizer, adapter
du -sh outputs/full_run/  # should be ~6GB
```

---

## Phase 4: Upload to HuggingFace (~10 min)

### 4.1 Upload the model
```bash
python scripts/upload_model.py --model-dir outputs/full_run --repo MenteEAI/MenteE-Code-3B
```

### 4.2 Upload the dataset (optional but recommended)
```bash
huggingface-cli upload MenteEAI/MenteE-Code-3B-Dataset data/training/train.jsonl train.jsonl
```

### 4.3 Verify on HuggingFace
```
https://huggingface.co/MenteEAI/MenteE-Code-3B
```

---

## Phase 5: Terminate Vast.ai (stop paying)

```bash
# On vast.ai dashboard: terminate instance
# Or via CLI:
vastai destroy instance <instance_id>
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