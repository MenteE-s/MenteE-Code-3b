# MenteE Code 3B — Executive Summary

> Status: **Planning / pre-data.** No training has been run. No benchmark numbers have been produced.
> Verified sources accessed **2026-10-19**. Every external claim in this document carries a source link and an access date.
> `📐` = assumption needing measurement · `✅` = verified from a primary source · `🎯` = our recommendation

---

## 1. What we are building

**MenteE Code 3B** is a code-specialised language model for **autonomous coding agents** (OpenCode and similar harnesses). It is:

- **3B activated parameters.** The "3B" is *active* parameters per token, not total. This is the industry-standard meaning in the MoE era and it is the only meaning under which "3B" is a competitive product claim.
- **Apache-2.0-compatible lineage**, so it can be served commercially without licence friction.
- **Served over an OpenAI-compatible `/v1/chat/completions` API**, so any OpenAI-compatible agent can use it — not just OpenCode.
- **Trained on our own high-quality, execution-verified agent trajectories**, which is the actual moat. The base model is a commodity; the data is not.

It is a **fine-tune of an existing open base**, not a from-scratch pretrain. This must be stated plainly in the model card (§1 of the source blueprint said "independently developed", which overstates it — see §7 below).

---

## 2. The one decision that determines everything else

**Base model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`** · **Teacher model: `Qwen/Qwen3-Coder-Next`**

| | Model | Total / Active | Licence | Context | Why this one |
|---|---|---|---|---|---|
| **Base (student)** | `Qwen3-Coder-30B-A3B-Instruct` | 30.5B / **3.3B** | Apache-2.0 ✅ | 262,144 ✅ | Standard attention stack → LoRA/QLoRA is a solved problem. **17.5 GB VRAM** for QLoRA → fits one 24 GB card. ✅ Serves at **18.6 GB** at Q4 → one 24 GB card. ✅ Native `qwen3_coder` tool-call parser. ✅ |
| **Teacher** | `Qwen/Qwen3-Coder-Next` | 80B / **3B** | Apache-2.0 ✅ | 262,144 ✅ | **70.6 SWE-bench Verified** ✅ — it *is* the "3B that works like a 300B". Apache-2.0 → **no teacher-terms ambiguity** for training on its outputs. ✅ Non-thinking mode, so trajectories are clean text + tool calls with no reasoning blocks to strip. ✅ |

**Why this pairing and not something else:**

1. **Both are Apache-2.0.** This is the single most important property. A commercial beta needs redistribution rights, and the training-data lineage needs to be clean. The alternative 3B-active model — `poolside/Laguna-XS-2.1` (33B/3B, 256K, 70.9 SWE-bench Verified ✅) — is licensed **OpenMDW-1.1 + Poolside AUP**, which is "fully permissive" ✅ but attaches an Acceptable Use Policy and carries explicit guidance about not circumventing safety guardrails without equivalent mitigations ✅. That is a compliance burden we do not need in v0.1.
2. **Same family, same tool-call serialisation.** Student and teacher share the `qwen3_coder` tool-call format ✅, so the student inherits format correctness *for free* and we only have to teach behaviour, not syntax. This removes the single most common failure mode of agentic fine-tunes (§6, Risk R3).
3. **Cost asymmetry is the whole business case.** 80B total / 3B active means the teacher generates frontier-adjacent trajectories at 3B inference cost. Generating 20k trajectories ≈ 240M output tokens ≈ **$290–900 of GPU time** instead of a four-figure API bill, and with zero licence ambiguity. 📐 See `09-COST-MODEL.md`.

**Fallback base (if 30B-A3B underperforms or a licence issue appears):** `poolside/Laguna-XS-2.1` (OpenMDW-1.1). **Stretch base (best quality, higher cost):** `Qwen3-Coder-Next` itself — but its Gated DeltaNet hybrid makes LoRA materially harder (§2.4).

### 2.1 The "3B that works like a 300B" — direct answer

`Qwen3-Coder-Next` is the literal answer: **80B total, 3B activated**, 262K context, **70.6 SWE-bench Verified / 44.3 SWE-bench Pro / 36.2 Terminal-Bench 2.0** (all three from the official Hugging Face eval results on the model card ✅). It is a hybrid `Gated DeltaNet + MoE + Gated Attention` model, 48 layers, hidden 2048, 512 experts with 10 active plus 1 shared ✅.

That architecture is the reason it can hold a 262K-token repo context: 12 blocks of 3 Gated DeltaNet layers (O(1) per token, fixed-size recurrent state) interleaved with 1 Gated Attention layer that periodically re-anchors global retrieval ✅.

### 2.2 The naming contract

> **"MenteE Code 3B" = 3B activated parameters per token.**

Because our base is 30.5B total / 3.3B active, "3B" is accurate *as an activation count*. The model card must state **both** numbers and the base model's identity. Anyone reading "3B" and assuming 3B total will be misled, and the disclosure is what keeps the claim honest. 🎯

### 2.3 Not the dense-3B path (and why)

There is a genuinely excellent **dense** 3B: `WeiboAI/VibeThinker-3B` — 3B dense, **MIT licence** ✅, built on Qwen2.5-Coder-3B ✅, **LiveCodeBench v6 80.2 Pass@1** ✅, 96.1% first-attempt LeetCode acceptance ✅. It is remarkable.

**We are not using it as the base**, because it is a *reasoning* model (AIME/olympiad focus, emits reasoning traces), not a tool-calling agent model, and reasoning-trace data is expensive to generate and to train on. It stays on the watchlist for a reasoning-flavoured v0.2. `Qwen3.8-27B` (27B dense, Apache-2.0, 73.0 Terminal-Bench 2.1 ✅) is the dense alternative if we ever want dense.

### 2.4 Known training traps on this base (read before writing the training script)

| Trap | Detail | Mitigation |
|---|---|---|
| **MoE LoRA rank is a different game** | For Qwen3-30B-A3B, expert intermediate dim `n=768` (not 25600 as in Qwen3-32B). At `r=64` that is `r/n = 0.083` — LoRA adapters are ~11% of each expert matrix ✅. Unsloth's own guidance suggests being careful here ✅. | Start at **r=16 on attention-only**, then r=32 all-linear. Validate that expert weights actually move. |
| **Router must stay frozen** | Fine-tuning a MoE router is destabilising. Unsloth disables it by default ✅. | Keep `modules_to_save` empty. Assert router grads are `None`. |
| **4-bit MoE import is broken upstream** | Unsloth: "The full 16-bit model must be downloaded and converted to 4-bit on the fly… This is due to issues importing 4-bit BnB MOE models directly" ✅ | Let the loader do the conversion. Don't try to import a pre-quantised MoE. |
| **16-bit LoRA needs 63 GB** | Qwen3-30B-A3B 16-bit LoRA = 63 GB VRAM ✅ | Use **QLoRA** (17.5 GB ✅) or offload (`experts4bit-qlora` reports 7.16 GB peak on a 12 GB card ✅). |
| **Gated DeltaNet LoRA (only if using Qwen3-Coder-Next)** | Copying a standard transformer's `target_modules` list silently trains only ~1/3 of the network — two-thirds of layers are Gated DeltaNet blocks ✅. Needs `flash-linear-attention` + `causal-conv1d` for acceptable speed (reported ~6–7× LoRA speedup ✅). | If we ever train Next: derive `target_modules` from `named_modules()` and **assert the gated-attention count is non-zero** before training. |
| **Our stack is too old** | `mentee-embeddings` pins `transformers==4.46.3` / `peft==0.14.0` ✅ (both from 2024). Qwen3-MoE needs newer `transformers`. | See `06-TRAINING.md` §2 for the exact pin set. |

---

## 3. The five gates

A release requires **all five**. No gate is substitutable for another.

| # | Gate | Threshold | Why this gate |
|---|---|---|---|
| **G1** | **Tool-call conformance** | ≥ 99.5% of emitted tool calls parse with the real OpenCode/vLLM `qwen3_coder` parser, across 500 held-out turns | A model the harness cannot parse scores zero on everything else. This is binary, not a score. |
| **G2** | **Agentic task completion** | MenteE Code 3B ≥ base model's completion rate on our 50-task in-harness suite, **with identical prompts** | A fine-tune that does not beat its own base has no product. |
| **G3** | **Cost per successful task** | ≤ **$0.10/task**, and ≥ 3× cheaper than the frontier-API comparison baseline | The entire commercial rationale. 3B wins on price, not on capability. |
| **G4** | **No regression on raw code** | LiveCodeBench / HumanEval+ drop ≤ 2 points absolute vs base | SFT on agentic data commonly *degrades* raw code ability. This gate catches that. |
| **G5** | **Licence + privacy + security sign-off** | Zero unresolved blocking items in `docs/11-RISKS-AND-OPEN-QUESTIONS.md` | Non-negotiable. |

### 3.1 The kill switch (written down on purpose)

**If G2 fails and no plausible prompt/data/base adjustment fixes it within 2 further experiments, we do not ship a fine-tune.** We ship:

1. the **base model** with our improved harness + prompt + context management, branded honestly, or
2. we re-scope to the **30B-A3B MoE at higher quality tiers** (a serving product, not a model product).

This is written into the plan so that it is a decision, not a negotiation. A 3B fine-tune that does not beat its own free base is a liability, not an asset. 🎯

---

## 4. The answer to "if we have more data, and the best quality data, are we good to go?"

**No. Not on its own. It is necessary, and it is roughly 60% of the outcome — but the other 40% is where projects like this actually die.**

More-and-better data gets you most of the way. Here is what it does **not** get you:

| Gap | What breaks | Why data volume does not fix it |
|---|---|---|
| **Base model choice** | Wrong base → you are teaching a model that cannot do the task | Worth more than the difference between 5k and 50k examples. A 70.6% SWE-bench teacher produces far better trajectories than a 20% one, *per example*. 🎯 |
| **Format conformance** | Model is fluent and the harness cannot parse it | A **serialisation bug**, not a data-volume problem. Fixed by reusing the base chat template byte-for-byte + a parser test. |
| **Verification** | Model learns to *look* successful | Unverified trajectories teach confident failure. Your weekend data must **actually run the tests**. This is the single highest-value thing you can do this weekend. 🎯 |
| **Held-out distribution** | Eval measures memorisation | If train and test come from the same repos and teacher, your score is a recall measurement, not a capability measurement. Split by **commit date**, not at random. |
| **Quantisation** | Post-training Q4 costs accuracy | A model that passes G2 at BF16 and fails at Q4 has not passed G2. Measure at serving precision. |
| **Context-length train/serve mismatch** | Trains at 8k, serves at 32k → silent quality loss | Agent trajectories are long. Train at the length you serve. 📐 |
| **The harness** | Great model, mediocre agent | The product is model + harness. Our harness (`MenteE SWE`, sibling repo) is half the value and we own it. |

**The formula we will use to decide:**

```
release quality  ≈  base_model  ×  data_quality  ×  verification  ×  format_conformance  ×  eval_discipline
```

Multiplicative, not additive. Any factor near zero makes the product near zero, and **no amount of extra data compensates for a zero in the others.** This is why the doc set spends as much on the base-model matrix, the schema, and the eval harness as it does on data volume.

### 4.1 The data-sizing answer, concretely

**For a 3B-activated model, 20,000 trajectories is a reasonable target — but it is the *quality floor* that matters, and more of the same kind has sharply diminishing returns.**

- 🎯 **Ship the 500-example experiment first.** If 500 verified trajectories do not beat the base on G2, 20,000 of the same will not either. You learn this in 3 days instead of 3 weeks.
- 🎯 **Coverage beats volume.** 20k trajectories across 200 repos in 4 languages beats 20k across 5 repos in 1 language, and the second one will silently overfit to those 5 repos.
- 📐 **Expected saturation band for a 3B:** roughly **50k–200k** high-quality trajectories. Past that, either diversify or move to RL — do not keep adding the same distribution. This is a reasoned estimate, not a measured one, and the v0.1 release should measure it rather than assume it.
- 🎯 **What to collect beyond volume:** *failure paths*. A trajectory that shows "test failed → read the error → patch → green" teaches debugging. A trajectory that shows only the final green patch teaches SFT. **Collect recoveries, not just successes.**

---

## 5. What we explicitly are not doing

| Not doing | Why |
|---|---|
| Pretraining from scratch | 3B from scratch is a multi-month, multi-thousand-GPU-hour project. We are bootstrapped. |
| Kubernetes / Kafka / distributed training | One 24 GB card trains the student. One 48 GB card serves the teacher. No measured requirement justifies more. |
| Closed-API teachers (Claude / GPT / Gemini) | Licence ambiguity for output use in training, plus a four-figure bill at 240M generated tokens. Apache-2.0 teachers remove both. |
| RFT and DPO in v0.1 | RFT first (we have a verifier → free labels). DPO needs preference pairs we'd have to fabricate. See `06-TRAINING.md` §5. |
| Any claim of benchmark improvement we have not measured | Non-negotiable, from the source blueprint. Every number in the model card will be a measured number or it will not be there. |

---

## 6. Top risks (full register in `11-RISKS-AND-OPEN-QUESTIONS.md`)

| ID | Risk | Sev | Mitigation |
|---|---|---|---|
| **R1** | Fine-tune does not beat its own base (G2 fails) | **High** | Kill switch (§3.1) is written down. Ship the base + harness if it fails. |
| **R2** | Data collection produces unverifiable trajectories | **High** | Sandbox executor is a Phase-1 deliverable, not a Phase-2 nicety. `05-DATA-COLLECTION-PLAYBOOK.md` §4. |
| **R3** | Tool-call format mismatch with the harness | **High** | Reuse base chat template byte-for-byte. G1 conformance test with the real parser, from Day 1. |
| **R4** | Training-data licence contamination | **High** | Source registry with a licence field is a **blocking gate**, not a document. |
| **R5** | Benchmark contamination from training repos | Med | Commit-date split; dedupe against benchmark repo lists; n-gram overlap check. |
| **R6** | DeltaNet / MoE fine-tuning pitfalls | Med | §2.4 mitigations; attention-only LoRA first. |
| **R7** | Beta cost blowout | Med | Per-key quota, hard spend cap, automatic kill switch, per-request cost accounting from day 1. |

---

## 7. The "independently developed" wording

The source blueprint described MenteE Code as "independently developed". That is true of the **post-training, data, evaluation and serving** — not of the **base weights**.

The model card must say: *"MenteE Code 3B is a QLoRA fine-tune of `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Apache-2.0), post-trained on MenteE's own execution-verified agent trajectory dataset."* That is a *stronger* claim than "independently developed", because it is specific and checkable. Compare with `mentee-embed`, which genuinely was trained from scratch and can say so. 🎯

---

## 8. Where to go next

| If you are | Read |
|---|---|
| **Collecting data this weekend** | `05-DATA-COLLECTION-PLAYBOOK.md` — this is your job, start here |
| **Writing the pipeline** | `04-DATA-SCHEMA.md` → `06-TRAINING.md` |
| **Choosing the base** | `02-BASE-MODEL-SELECTION.md` |
| **Deciding what "good" means** | `07-EVALUATION.md` |
| **Shipping** | `08-SERVING-AND-BETA.md` |
| **Paying for it** | `09-COST-MODEL.md` |
| **Executing** | `10-ROADMAP-AND-BACKLOG.md` |
| **Worried** | `11-RISKS-AND-OPEN-QUESTIONS.md` |

---

*All external facts verified against primary sources on 2026-10-19. See `12-OPEN-QUESTIONS.md` for what was **not** verified.*
