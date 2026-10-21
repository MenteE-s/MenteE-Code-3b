# Base Model & Teacher Selection

> All external facts verified against primary sources on **2026-10-19**. `✅` verified · `📐` assumption · `🎯` recommendation
> Machine-readable candidate matrix: `configs/base_model_matrix.json`

---

## 1. Decision

```
PRIMARY BASE   Qwen/Qwen3-Coder-30B-A3B-Instruct      Apache-2.0
FALLBACK BASE  poolside/Laguna-XS-2.1                 OpenMDW-1.1
STRETCH BASE   Qwen/Qwen3-Coder-Next                  Apache-2.0
PRIMARY TEACHER Qwen/Qwen3-Coder-Next                 Apache-2.0
DENSE ALT      Qwen/Qwen3.8-27B                       Apache-2.0
REASONING ALT  WeiboAI/VibeThinker-3B                 MIT
```

**The base and the teacher are deliberately different models.** The teacher is stronger (70.6 vs a much lower SWE-bench Verified for 30B-A3B); the base is *easier to fine-tune and cheaper to serve*. That asymmetry is the point: we pay 3B-active inference cost to generate frontier-adjacent trajectories, then bake their behaviour into a 3B-active student that costs 3B-active inference cost to serve.

---

## 2. Candidate matrix

### 2.1 3B-activated MoE tier (the tier that matches "3B that works like 300B")

| | **Qwen3-Coder-30B-A3B-Instruct** | **Qwen3-Coder-Next** | **Laguna XS 2.1** | **Laguna S 2.1** |
|---|---|---|---|---|
| Total / active | 30.5B / **3.3B** ✅ | 80B / **3B** ✅ | 33B / **3B** ✅ | 118B / **8B** ✅ |
| Licence | **Apache-2.0** ✅ | **Apache-2.0** ✅ | OpenMDW-1.1 (+ AUP) ✅ | OpenMDW-1.1 (+ AUP) ✅ |
| Commercial use | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| Context (native) | **262,144** ✅ | **262,144** ✅ | 256K ✅ | 1,048,576 ✅ |
| Architecture | Qwen3-MoE, 48L, h=2048, GQA 32Q/4KV, 128 experts / 8 active ✅ | Hybrid: 12×(3×GatedDeltaNet→MoE + 1×GatedAttention→MoE), 48L, h=2048, 512 experts / 10 active + 1 shared ✅ | 33B-A3B MoE ✅ | 118B-A8B MoE, 256 experts / 10 active ✅ |
| SWE-bench Verified | not published here | **70.6** ✅ | — (SWE-bench Multilingual **63.1**) ✅ | — |
| SWE-bench Pro | — | **44.3** ✅ | — | — |
| Terminal-Bench 2.0 | — | **36.2** ✅ | — | — |
| Thinking mode | yes ✅ | **no** ✅ (non-thinking only) | **yes, requires preserving reasoning in history** ✅ | yes ✅ |
| Tool-call format | `qwen3_coder` ✅ | `qwen3_coder` ✅ | native FC ✅ | native FC ✅ |
| QLoRA VRAM | **17.5 GB** ✅ (Unsloth) | ~40 GB est 📐 | 📐 | 📐 |
| Serve VRAM @ Q4 | **18.6 GB** ✅ → 1×24 GB card | ~45 GB 📐 → 1×48 GB card | ~20 GB 📐 | ~70 GB ✅ |
| Fine-tuning maturity | **high** — Unsloth supports 30B-A3B ✅ | low — Gated DeltaNet LoRA pitfalls ✅ | low | low |
| **Verdict** | **PRIMARY BASE** | **PRIMARY TEACHER** | **FALLBACK BASE** | serving-only |

### 2.2 Dense candidates (for a future dense-3B line)

| | **VibeThinker-3B** | **Qwen3.8-27B** | **Qwen2.5-Coder-3B** |
|---|---|---|---|
| Params | **3B dense** ✅ | 27B dense ✅ | 3B dense ✅ |
| Licence | **MIT** ✅ | **Apache-2.0** ✅ | Apache-2.0 ✅ |
| LiveCodeBench v6 | **80.2** ✅ | — | — |
| Terminal-Bench 2.1 | — | **73.0** ✅ | — |
| AIME26 / IMO-AnswerBench | **94.3 / 76.4** ✅ | 48.7 (IMO-AB, Qwen3.5-4B) ✅ | — |
| LeetCode 1st-attempt accept | **96.1%** ✅ | — | — |
| Build base | Qwen2.5-Coder-3B ✅ | — | — |
| Gotcha | reasoning model, not tool-calling | **hybrid attention + natively multimodal** → `from_pretrained` returns a *processor*, and a standard `target_modules` list trains only ~1/3 of the network ✅ | older, weaker |
| **Verdict** | watchlist for v0.2 reasoning | dense alt | legacy |

### 2.3 Frontier API teachers (rejected for v0.1)

| Teacher | Why rejected |
|---|---|
| Claude / GPT / Gemini | Output-use terms for training a competing model need a legal read; 240M output tokens at frontier rates is a four-figure bill. Both problems are solved by an Apache-2.0 self-hosted teacher. |
| Kimi K2.6 / K2.7 Code | MIT-modified licence ✅, but closed weights → same API-bill problem, and licence reading is non-trivial. |
| DeepSeek V4.1 Flash (off-peak API) | Same API-bill problem. |

**The generalisation:** a self-hosted Apache-2.0 teacher is *both* cheaper *and* legally simpler. That is not a coincidence — it is the same fact seen twice.

---

## 3. Licence checks (blocking, not advisory)

| Check | 30B-A3B | Coder-Next | Laguna XS 2.1 |
|---|---|---|---|
| Permits commercial use | ✅ Apache-2.0 | ✅ Apache-2.0 | ✅ OpenMDW-1.1 described as fully permissive ✅ |
| Permits modification / fine-tuning | ✅ | ✅ | ✅ |
| Permits redistribution of derivatives | ✅ | ✅ | ✅ |
| Requires attribution to base | ✅ (Apache-2.0 §4) | ✅ | ✅ check AUP |
| Attaches an AUP | ❌ none | ❌ none | ✅ **Poolside AUP** + "advise against circumventing safety guardrails without implementing substantially equivalent mitigations" ✅ |
| Adds non-Apache obligations | ❌ | ❌ | ✅ yes — accept or implement mitigations |
| **Verdict for a commercial beta** | **clean** | **clean** | usable, but carries an AUP and a guardrail duty |

🎯 **Apache-2.0 lineage is a hard requirement for v0.1.** It is the only licence pair here with no attached use policy.

**Apache-2.0 obligations we must honour in the model card:** retain the copyright/attribution notice, state changes made, and include the licence text. We will additionally state the base model id, the base licence, and the fact that the weights are a derivative. ✅

**Teacher-output note:** Apache-2.0 governs the *weights*. It does not formally govern *outputs*. In practice, using an Apache-2.0 model's outputs as training data is uncontroversial because the output is not a derivative of the licensed material and carries no restriction. We will still record provenance for every trajectory (teacher id, revision, sampling params, licence) so the lineage is auditable. 🎯

---

## 4. Architecture consequences

### 4.1 Student = Qwen3-Coder-30B-A3B-Instruct

**Chat template & tool-call serialisation: reuse the base's, byte for byte.** Not "similar" — identical. The `qwen3_coder` tool-call format is already what vLLM's `--tool-call-parser qwen3_coder` parses and what OpenCode expects ✅. Any divergence means the harness sees unparseable output, and the model looks incompetent while being perfectly capable.

> **Hard rule:** the data pipeline renders training text using `tokenizer.apply_chat_template()` from the base repo, with `tools=` passed in. A hand-written renderer is a bug waiting to happen. The conformance test in `06-EVALUATION.md` §3 is the enforcement.

**Sampling (serving, from the model card):** `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05` ✅ — note this differs from Coder-Next's `temperature=1.0, top_p=0.95, top_k=40` ✅. Do not mix them up between teacher and student.

### 4.2 Teacher = Qwen3-Coder-Next

**Non-thinking mode only** ✅ — this is a gift. Teacher trajectories are clean text + tool calls with no `<think>` blocks to strip, no reasoning-preservation logic, and no risk of the student learning to emit reasoning tokens that the harness cannot route.

**Serving the teacher:** needs ~45 GB at Q4 📐 → one 48 GB card (RTX A6000 ~$0.43/hr ✅ / L40S ~$0.47/hr ✅). vLLM ≥ 0.15.0 or SGLang ≥ 0.5.8 ✅.

```bash
# Teacher serving (48 GB card, Q4)
vllm serve Qwen/Qwen3-Coder-Next \
  --port 8000 --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --quantization awq --max-model-len 32768
```
> `--max-model-len 32768` is the card's own OOM advice ✅. Start there; raise only if the trajectory length distribution says you must.

### 4.3 Why the student is not the teacher

| | Teacher (Coder-Next) | Student (30B-A3B) |
|---|---|---|
| Fine-tune | Gated DeltaNet LoRA is hard: a standard `target_modules` list trains only ~1/3 of the network ✅; needs `flash-linear-attention` + `causal-conv1d` for ~6–7× speed ✅ | Standard attention → LoRA is a solved problem ✅ |
| QLoRA VRAM | ~40 GB 📐 | **17.5 GB** ✅ |
| Serve VRAM @ Q4 | ~45 GB 📐 | **18.6 GB** ✅ |
| Serving cost for the beta | 48 GB card | **24 GB card** |
| Ability to add a modality later | hybrid arch, more brittle | cleaner |

The teacher's job is to produce 20k good trajectories. The student's job is to be cheap to serve forever. Using the same model for both would make the beta cost 2× for no benefit.

### 4.4 MoE-specific training rules

- **Freeze the router.** Unsloth disables router training by default ✅. Assert router gradients are `None` — a silently-trained router is a classic way to destroy an MoE.
- **LoRA rank must account for expert width.** For Qwen3-30B-A3B: `m=2048`, expert intermediate `n=768`, `E=128` experts, `k=8` active ✅. At `r=64` each adapter adds `r(m+n)=180,224` params per expert ≈ **11% of the expert matrix** ✅. That is large. 🎯 Start **r=16, attention-only** (`q_proj,k_proj,v_proj,o_proj`) → then r=32 all-linear. Validate that expert weights actually move before trusting the run.
- **4-bit MoE import is broken.** The full 16-bit model must be downloaded and converted on the fly ✅. Do not try to `from_pretrained` a pre-quantised MoE.
- **16-bit LoRA needs 63 GB** ✅. Not viable on a 24 GB card. Use QLoRA, or `experts4bit-qlora` (7.16 GB peak on a 12 GB card ✅) if we need long sequences.

---

## 5. Serving precision decision

| Precision | VRAM @ 30B-A3B | Notes |
|---|---|---|
| BF16 | ~61 GB | Reference only. Too big for a 24 GB card. |
| FP8 | ~31 GB | Needs 40 GB+ card. Better quality than AWQ. |
| **AWQ / GPTQ Q4** | **18.6 GB** ✅ | **Default.** Fits one 24 GB card, leaving headroom for KV cache. |

🎯 **Measure every gate at Q4**, not at BF16. A model that passes G2 at BF16 and fails at Q4 has not passed G2. This is the single most common way agentic models are over-claimed. See `06-EVALUATION.md` §2.

**Context length served:** start at **32,768** and only raise it if the beta's measured context distribution requires it. Serving 262K on a 24 GB card spends nearly all VRAM on KV cache, crushing concurrency for no measured benefit. 📐

---

## 6. Replacement plan

Every choice here is designed to be swappable:

| If this changes | Cost to swap |
|---|---|
| Base model | Re-render the dataset with the new base's chat template (one function), retrain (~6 GPU-hours), re-run gates. **Data is preserved.** |
| Teacher model | Regenerate trajectories (GPU-hours). The schema does not change. |
| Serving precision | Re-quantise, re-run G1–G4. No retraining. |
| Licence of the base | Only if the fallback is needed; the fallback is already licences-checked. |

The **only** irreversible decision here is the data, and even that is preserved across base-model changes by the schema design in `03-DATA-SCHEMA.md`.

---

## 7. Sources

| Claim | Source | Accessed |
|---|---|---|
| Qwen3-Coder-30B-A3B specs, licence, sampling, tool-call format | [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 2026-10-19 |
| Qwen3-Coder-Next specs, architecture, evals (70.6 / 44.3 / 36.2), vLLM/SGLang requirements, sampling | [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 2026-10-19 |
| Laguna XS 2.1 — 33B/3B, 256K, OpenMDW-1.1, AUP, SWE-bench Multilingual 63.1, thinking requirements | [HuggingFace model card](https://huggingface.co/poolside/Laguna-XS-2.1) · [Poolside blog](https://poolside.ai/blog/introducing-laguna-xs-2-1) | 2026-10-19 |
| Laguna S 2.1 — 118B/8B, 1M ctx, 70.2 Terminal-Bench 2.1, OpenRouter pricing | [HPCwire](https://www.hpcwire.com/aiwire/2026/08/24/poolside-launches-laguna-s-2-1-open-weight-coding-model) | 2026-10-19 |
| VibeThinker-3B — 3B dense, MIT, LCBv6 80.2, AIME26 94.3, LeetCode 96.1% | [HuggingFace model card](https://huggingface.co/WeiboAI/VibeThinker-3B) · [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1) | 2026-10-19 |
| Unsloth — 30B-A3B QLoRA 17.5 GB, 16-bit LoRA 63 GB, router frozen, MoE support, 4-bit import caveat | [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune) · [Faster MoE](https://unsloth.ai/docs/basics/faster-moe) | 2026-10-19 |
| Gated DeltaNet LoRA target-module pitfall + `flash-linear-attention` speedup | [Medium: Fine-Tuning Qwen3.8-27B](https://medium.com/@ishaafsalman/fine-tuning-qwen3-8-27b-what-breaks-and-how-to-fix-it-d77dc46e0745) · [NVIDIA DGX Spark thread](https://forums.developer.nvidia.com/t/lossless-7-67x-lora-8-35x-full-ft-speedup-for-qwen3-5-on-dgx-spark-gb10-sm_121a/370517) | 2026-10-19 |
| `experts4bit-qlora` 7.16 GB peak, bit-identical forward | [Unsloth discussion #3163](https://github.com/unslothai/unsloth/discussions/3163) | 2026-10-19 |
| Laguna S 2.1 available on OpenCode | [kie.ai](https://kie.ai/blog/what-is-laguna-s-2-1) | 2026-10-19 |
