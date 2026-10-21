# Training Methodology & Initial Configuration

> Base: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Apache-2.0) ✅ · Teacher: `Qwen/Qwen3-Coder-Next` (Apache-2.0) ✅
> Everything here is **config-as-code** and resumable, matching the `mentee-embeddings`
> repo's `configs/ → scripts/ → reports/ → models/` pattern ✅.

---

## 1. The three stages

| Stage | Data | Purpose | Cost 📐 | Gate to advance |
|---|---|---|---|---|
| **Stage 0 — smoke** | 20 examples | Prove the pipeline end to end. Nothing else. | ~$1 | 20 steps run, loss finite, checkpoint saves & reloads |
| **Stage 1 — pilot** | **500 verified** | 🎯 **The experiment that decides the project.** | ~$5–15 | **G2** (in-harness) + **G4** (no regression) |
| **Stage 2 — scale** | 2k → 20k | The release model. | ~$50–250 | G1–G4 all pass |
| **Stage 3 — RFT** | self-generated, verifier-filtered | On-policy specialisation | ~$30–150 | G2 improves again |

> 🎯 **Stage 1 is the whole project in miniature.** If 500 verified examples do not beat the
> base model, 20,000 will not. Run it in 3 days, not 3 weeks.

---

## 2. Environment

### 2.1 The version bump you need

Your `mentee-embeddings` stack is pinned to `transformers==4.46.3`, `peft==0.14.0`,
`accelerate==1.0.1` ✅ — all from 2024. **Qwen3-MoE needs newer `transformers`.**

`requirements-train.txt` for this project (📐 verify against the current release before the run):

```
torch==2.7.1                 # keep — already cu128 / sm_120 (RTX 5090) ✅
transformers>=4.51,<5        # Qwen3-MoE support
tokenizers>=0.21
peft>=0.14                   # 0.14 is probably fine, but re-verify
accelerate>=1.0
bitsandbytes>=0.45           # 4-bit
datasets>=3.0
trl>=0.12                    # only if using TRL SFTTrainer
unsloth                      # stage 0/1 for VRAM + speed; optional
```

> ⚠️ **Do not blindly reuse `mentee-embeddings`' `requirements-vast.txt`** — it pins
> `transformers==4.46.3` ✅ which predates Qwen3-MoE. Copy `setup_vast.sh` ✅ (the GPU
> detection, the cu128 torch install, the venv bootstrap) and replace only the deps file.

### 2.2 Hardware

| Stage | GPU | Why |
|---|---|---|
| Teacher serving | 48 GB (RTX A6000 / L40S) | ~45 GB at Q4 📐 for the 80B teacher |
| Stage 0/1 training | **24 GB (RTX 4090 / 5090)** | QLoRA fits in **17.5 GB** ✅ |
| Stage 2 long-context training | 48 GB (A6000 / L40S) | 32K sequences need the headroom |
| Student serving | 24 GB | 18.6 GB at Q4 ✅ |

**One 24 GB card is enough for the entire pilot.** No multi-GPU, no FSDP, no DeepSpeed. 🎯

---

## 3. Stage 0 — smoke test (must pass before anything else)

**Purpose: prove the plumbing. Not to produce a model.**

```bash
python -m src.train.sft \
  --config configs/train_config.yaml \
  --override data.max_examples=20 \
  --override train.max_steps=20 \
  --override train.save_steps=10 \
  --override output.run_name=smoke
```

**Acceptance criteria — all five, or nothing ships:**

- [ ] 20 steps complete without OOM
- [ ] loss is finite and trending down (expect ~2.5 → ~1.5 on 20 examples — a rough sanity check, not a target)
- [ ] the adapter checkpoint saves and reloads into a fresh process
- [ ] the reloaded model produces a **parseable tool call** for a trivial prompt
- [ ] the run is **resumable**: kill it at step 15, restart, and it continues from 15

> 🎯 The fifth criterion is the one people skip, and it is the one that saves you when a
> 6-hour run dies at hour 5. Test resume *in the smoke test*, not in production.

### 3.1 The tokenizer/trap check (do this first, it takes 2 minutes)

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")

msgs = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the bug."},
]
rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print(rendered)
```

**Assert that `rendered` contains the tool-call scaffolding the model card documents**
(`<tool_call>` / `</tool_call>` tokens ✅) and that the same renderer is used for training *and*
for serving-time verification. 🎯

If the training renderer and the serving renderer differ by even one token, the model will
be fine in your eval harness and broken in the real one — and you will lose a day to a bug
that took five minutes to catch here.

---

## 4. Stage 1 — the pilot (500 examples)

### 4.1 Initial hyperparameters

`configs/train_config.yaml` — starting point, tune on the pilot:

```yaml
run_name: mentee-code-3b-pilot-01
seed: 42

model:
  base: Qwen/Qwen3-Coder-30B-A3B-Instruct
  base_revision: <pin a commit sha, not 'main'>
  quantization: bnb_4bit          # QLoRA
  attn_implementation: flash_attention_2
  load_in_4bit: true
  use_gradient_checkpointing: true   # required at 16K seq

lora:
  r: 16                          # START LOW — see §4.2
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
  target_modules:                # PHASE 1: attention only
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  # PHASE 2 (only if attention-only was insufficient): add
  #   - gate_proj
  #   - up_proj
  #   - down_proj
  modules_to_save: []            # router MUST stay frozen

data:
  dataset_version: v1.0.0
  train: data/processed/v1.0.0/train.jsonl
  val:   data/processed/v1.0.0/val.jsonl
  max_seq_len: 16384             # train at what you serve
  truncation: error              # never silently truncate
  loss_on: assistant_only        # tool-call tokens must be supervised

train:
  micro_batch_size: 1
  grad_accum: 16                 # effective batch 16
  lr: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_grad_norm: 1.0
  epochs: 2
  precision: bf16
  optim: adamw_8bit
  max_steps: null

checkpoint:
  dir: models/runs/mentee-code-3b-pilot-01
  save_steps: 50
  save_total_limit: 3
  resume_from: auto               # crash-safe

eval:
  every_steps: 50
  metric: eval_loss

output:
  hf_repo: MenteEAI/mentee-code-3b-pilot-01
  base_model_id: Qwen/Qwen3-Coder-30B-A3B-Instruct
  base_model_license: Apache-2.0
  dataset_version: v1.0.0
```

### 4.2 Why `r=16` and attention-only first

For Qwen3-30B-A3B the expert intermediate dimension is `n=768` ✅ (not 25600 as in a dense
Qwen3-32B). At `r=64` a LoRA adapter adds ~11% of each expert matrix ✅ — that is a very
large perturbation for a 3B-active model trained on 500 examples.

> 🎯 **Start with attention-only at r=16.** If the pilot shows the model is not learning
> the *behaviour* (it will have learned the format from the base already), escalate to r=32
> all-linear. Escalating straight to r=64 all-linear on 500 examples is the most common way
> to waste the pilot.

### 4.3 Loss masking — the thing that silently breaks SFT

**Supervise only the assistant turns** (including the `tool_call` tokens). Supervising the
tool *results* teaches the model to hallucinate tool output. 🎯

```
system / user / tool_result  →  loss = 0   (masked)
assistant (incl. tool_calls) →  loss = CE   (supervised)
```

**Test this before training.** Render one example, print the token ids with their loss mask,
and confirm the tool-result tokens are `-100`. A single unmasked tool-result span teaches
the model to fabricate test output, and it will look like a "model quality" problem for
weeks. 🎯

### 4.4 Sequence length and truncation

| Setting | Value | Reason |
|---|---|---|
| `max_seq_len` | **16,384** | Matches the serving target (`08-SERVING-AND-BETA.md`). Train/serve mismatch is a silent quality tax. |
| Truncation | **`error`** | A truncated trajectory teaches the model the trajectory *ends there* — it learns to stop mid-task. |
| Dropped long examples | log the count | If > 10% exceed 16K, the distribution is wrong; fix the data, don't truncate. |

> 📐 The pilot's actual turn-length distribution is unknown until you measure it. Measure it
> in the first 20 examples and set `max_seq_len` from the data, not from a guess.

### 4.5 Reproducibility

Every run records: dataset version, base model id + revision, git SHA of the config, full
config dump, seed, and the resolved package versions. A run without all of these is not
reproducible and is not a release. 🎯

```
models/runs/<run_name>/
├── config.resolved.yaml
├── env.txt            # pip freeze, nvidia-smi, python version
├── git.sha
├── metrics.jsonl
├── checkpoints/
└── README.md
```

---

## 5. Stage 3 — RFT (rejection fine-tuning), before DPO

### 5.1 Why RFT, not DPO

> 🎯 **We have a verifier. A verifier gives labels for free. DPO needs preference pairs we
> would have to fabricate — and fabricated preference pairs are a well-known way to
> degrade a model while believing you improved it.**

RFT loop:

```
1. Current student generates N attempts per training task (N = 4–8, temperature 0.9)
2. Sandbox verifies each attempt
3. Keep the attempts that PASS (and the highest-scoring failures as recovery examples)
4. Add to the dataset as a new version
5. Fine-tune again from the previous adapter
6. Repeat 1–3 times
```

**Why this is the right second stage:** it is **on-policy** (the data is what the model
actually does, not what a teacher would do), and every example is **verified by
construction**. Both of the things the blueprint asked for — "separates SFT from true
logit-based or on-policy distillation" and "optional preference optimisation after the
initial model has been validated" — are satisfied by RFT, and DPO is deferred. 🎯

**Cost control:** use the *pilot's* 500 tasks, not 20k. RFT on 500 tasks × 8 attempts is
4,000 generations ≈ a few GPU-hours, and it is the highest-value few GPU-hours in the
project.

### 5.2 The stage-3 data rule

**Only add attempts the model actually produced.** Never mix RFT data with teacher data in
the same run — the distribution shift confounds the measurement. One stage, one data
version, one number. 🎯

---

## 6. Training commands (reproducible)

```bash
# 0. provision (reuse mentee-embeddings' setup_vast.sh ✅)
bash scripts/setup_vast.sh
source .venv/bin/activate

# 1. preflight — env, GPU, disk, dataset presence, schema validation
python -m src.preflight

# 2. stage 0 — smoke
python -m src.train.sft --config configs/train_config.yaml \
  --override data.max_examples=20 --override train.max_steps=20 \
  --override output.run_name=smoke

# 3. stage 1 — pilot (500)
python -m src.train.sft --config configs/train_config.yaml \
  --override output.run_name=pilot-01

# 4. evaluate
python -m src.eval.run --adapter models/runs/pilot-01/final --suite g1,g2,g4

# 5. stage 3 — RFT
python -m src.train.rft --config configs/rft_config.yaml
```

**`src/preflight` is mandatory before every run** — it must check GPU VRAM, dataset
existence, **schema validation of the dataset**, base model availability, and free disk
space, and exit non-zero on any failure. ✅ (`mentee-embeddings` already has a
`preflight.sh` ✅ — copy the pattern.)

---

## 7. Hardware & time estimates

| Stage | GPU | Wall-clock 📐 | Cost 📐 |
|---|---|---|---|
| Teacher serving (for data gen) | 48 GB | continuous | $0.43–0.47/hr |
| Stage 0 smoke | 24 GB | ~10 min | < $0.10 |
| **Stage 1 pilot (500 ex, 16K seq, 2 ep)** | 24 GB | **3–6 hrs** | **$1–4** |
| Stage 2 scale (20k ex, 16K seq, 2 ep) | 48 GB | 12–30 hrs | $6–20 |
| Stage 3 RFT (500 tasks × 8 attempts) | 24 GB + teacher | 4–8 hrs | $3–8 |
| Evaluation (G1–G4) | 24 GB | 2–4 hrs | $1–3 |

> 📐 **These are planning estimates, not measurements.** The blueprint explicitly forbids
> inventing training durations before profiling a representative subset. **Measure Stage 0's
> real throughput (tokens/sec) and multiply.** That is the whole point of Stage 0 existing.
>
> Rates from `09-COST-MODEL.md`, verified 2026-10-19.

---

## 8. What "good" looks like for the pilot

| Signal | Good | Worry |
|---|---|---|
| eval_loss | Drops ~15–30% over 2 epochs on 500 examples | Flat → the data isn't teaching; or loss is masking wrong |
| Tool-call format | ≥ 99.5% parse rate after 50 examples | Any parse failure → go back to §3.1 |
| Recovery behaviour | Model retries after a tool error instead of looping | Identical repeated tool calls → the anti-loop guard matters |
| Honest reporting | Says "the test still fails" when it does | Claims success on a red test → **reject the model, not just the data** |
| Base comparison | ≥ base on G2, ≤ 2 pts drop on G4 | G2 below base → kill switch, see `00-EXECUTIVE-SUMMARY.md` §3.1 |

---

## 9. Publishing

```bash
# merge the adapter into the base and quantise
python -m src.export.merge_and_quantize \
  --base Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --adapter models/runs/pilot-01/final \
  --out models/MenteE-Code-3B-pilot-01-AWQ \
  --quant awq

# validate the artefact before pushing
python -m src.export.validate --model models/MenteE-Code-3B-pilot-01-AWQ

# push (uses the mentee-embeddings' upload_hf.py pattern ✅)
python -m src.export.push_hf --model models/MenteE-Code-3B-pilot-01-AWQ \
  --repo MenteEAI/mentee-code-3b-pilot-01
```

**The model card must state:** base model id + revision, base licence (Apache-2.0 ✅),
the dataset version, the training config, the **measured** results (not estimates), the
known limitations, and the intended use. 🎯

> ⚠️ **Apache-2.0 §4 requires retaining the base model's copyright/attribution notice and
> stating that changes were made.** A model card that omits this is a licence violation, not
> a documentation nit. ✅
