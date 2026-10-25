---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
language:
- en
- code
tags:
- code
- agentic
- lora
- sft
- mentee
library_name: transformers
---

# MenteE Code 3B (v1)

**MenteE Code** is a coding-agent assistant built by **MenteE**. This is v1: a LoRA
fine-tune of [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
(Apache-2.0), merged into a standalone model and trained to behave as MenteE Code.

Ask it who it is and it will tell you: *"I am MenteE Code, a coding assistant built by MenteE."*
The chat template injects the MenteE identity by default, so every client
(transformers, vLLM, llama.cpp, Ollama) presents it correctly.

![HumanEval comparison](humaneval_compare.png)

## Training (all numbers measured)

| Item | Value |
|---|---|
| Base | Qwen2.5-Coder-3B-Instruct (3.09B params) |
| Data | 53,567 schema-validated agent trajectories (deduplicated from 78,212) |
| Sources | SWE-agent, OpenHands, R2E-Gym, NVIDIA SWE-Zero subsets + MenteE collector + identity Q&A |
| Method | LoRA r=8, α=16, all-linear targets, bf16, gradient checkpointing |
| Schedule | 1,250 steps · 1 epoch · effective batch 16 · lr 2e-4 · seq len 4,096 |
| Context | 32,768 native · **131,072 via YaRN** (factor 4.0, set in config.json) |
| Hardware | 1× RTX 5090 (32 GB) · 5 h 32 m · ~$0.13 |
| Final | loss 1.62 → **0.317** · token accuracy **91.5%** |

![Training loss](loss_curve.png)

![Token accuracy](accuracy_curve.png)

![Learning rate](lr_schedule.png)

## Benchmarks — measured, not estimated

HumanEval pass@1, greedy decoding, 50 problems, both models run on identical
hardware with an identical harness (generated code executed against the official
unit tests):

| Model | pass@1 |
|---|---|
| Qwen2.5-Coder-3B-Instruct (base) | **94.0%** |
| MenteE Code 3B v1 | **50.0%** |

**We publish this gap on purpose.** Failure analysis shows 24 of 25 v1 failures are
the same artifact — systematically doubled closing brackets — caused by v1's training
recipe (loss applied to all trajectory roles, including noisy tool outputs).
It is a data-recipe issue, not a capability ceiling. The v2 recipe (assistant-only
loss, identity oversampling, LoRA r=16) targets exactly this.

This is a 50-problem subset (~±14 pt margin) — directional, not leaderboard-grade.

![GPU telemetry](gpu_telemetry.png)

![Model landscape](model_landscape.png)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

t = AutoTokenizer.from_pretrained("MenteEAI/MenteE-Code-3B")
m = AutoModelForCausalLM.from_pretrained("MenteEAI/MenteE-Code-3B", torch_dtype="auto", device_map="auto")
msgs = [{"role": "user", "content": "Write a Python function to reverse a linked list."}]
prompt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = t(prompt, return_tensors="pt").to(m.device)
out = m.generate(**enc, max_new_tokens=512)
print(t.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

GGUF builds (for Ollama / llama.cpp) are published at
[MenteEAI/MenteE-Code-3B-GGUF](https://huggingface.co/MenteEAI/MenteE-Code-3B-GGUF).

## Roadmap

- **v2:** assistant-only loss, identity oversampling, LoRA r=16 → recover base-level coding quality
- **Flywheel:** real user conversations are logged (opt-in), converted to training data, and folded into future versions
- **Eval:** SWE-bench Verified subset with a proper agent harness

## License & attribution

Apache-2.0, same as the base model. Built on Qwen2.5-Coder-3B-Instruct by the
Qwen team — with gratitude. MenteE is not affiliated with Alibaba or Qwen.
