# The Thesis: How a 3B Model Behaves Like a 300B Model

> This is the load-bearing idea of MenteE Code 3B. If we get this wrong, the project is
> a downgrade of a free public model. If we get it right, we have a product.
>
> Verified against primary sources **2026-10-19**. `✅` verified · `📐` assumption · `🎯` recommendation

---

## 1. Confirmed: the tier is real, and it is demonstrated

A model with **3 billion activated parameters** is publicly documented reaching scores that
300B-class dense models reach on agentic coding tasks. This is not theoretical.

| Model | Total / Active | Licence | Agentic evidence (primary source) |
|---|---|---|---|
| **Qwen3-Coder-Next** | 80B / **3B** ✅ | **Apache-2.0** ✅ | **SWE-bench Verified 70.6** ✅ · **SWE-bench Pro 44.3** ✅ · **Terminal-Bench 2.0 36.2** ✅ |
| **poolside/Laguna-XS-2.1** | 33B / **3B** ✅ | OpenMDW-1.1 ✅ | **SWE-bench Multilingual 63.1** ✅ · 256K context ✅ |
| **WeiboAI/VibeThinker-3B** | **3B dense** ✅ | **MIT** ✅ | **LiveCodeBench v6 80.2** ✅ · 96.1% first-attempt LeetCode ✅ · AIME26 94.3 ✅ |

All three figures above for Qwen3-Coder-Next are the model's **own official Hugging Face eval
results** ✅ — not a blog claim.

> **Confirmed.** "3B that works like a 300B" is a demonstrated, licensed, shipping reality.
> It is not a stretch goal we are inventing to justify the project.

**But — and this is the part that decides our design — it is not a property of "3B".**
It is a property of a specific stack of five mechanisms. Get three of the five and you have
a mediocre 3B model. Get all five and you have a 300B-class one.

---

## 2. The five mechanisms (the causal chain)

### M1 — Sparse knowledge store: total ≫ active

**The single biggest reason.** A MoE with 80B total parameters and 10-of-512 experts active
per token ✅ means a 3B *compute* budget can reach into an 80B-sized *knowledge* pool. The
router performs the retrieval.

This is the mechanism that does not exist for a dense 3B. A dense 3B **is** its knowledge —
there is no larger pool to draw on.

> **Consequence:** 300B-class behavior at 3B active is largely **inherited from the base
> model's total parameter count**, not earned by our training. If we pick a base with a small
> total pool, we cap our ceiling no matter how good the data is.

### M2 — Cheap long context

Coder-Next's architecture is `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` ✅.
Gated DeltaNet is O(1) per token with a fixed-size recurrent state; the interleaved Gated
Attention layer periodically re-anchors global retrieval ✅.

**This is what lets a 3B-active model hold a 262,144-token repository** ✅ without the quadratic
cost that would otherwise blow the budget. A dense 3B with standard attention degrades long
before that.

> **Consequence:** a "300B-like" model is not just a better reasoner — it is a model that can
> *hold* the repo in context. If we train at 8k and serve at 32k, we have quietly broken M2.

### M3 — Execution-verified post-training (RL)

Qwen describes Coder-Next's training as "large-scale **executable task synthesis + RL**" ✅.
Poolside describes Laguna's as "**async off-policy agent RL**" ✅.

This is the mechanism that separates *"generates plausible code"* from *"completes the task and
knows when it failed."* Reinforcement against a **verifier** — did the test go green — is the
only training signal that produces genuine task completion rather than imitation of it.

> **Consequence:** this is the mechanism our data collection is *directly* for. Verified
> trajectories are the seed for this signal. Unverified trajectories only ever teach
> imitation.

### M4 — Long-horizon trajectory training

The model must be trained on **the whole agent loop**, including the parts where things go
wrong: tool error → read the error → revise → retry → recover.

A trajectory containing only the successful final patch teaches supervised fine-tuning. A
trajectory containing the failures *and* the recovery teaches an agent. 🎯

> **Consequence:** this is the second mechanism our data collection is for, and it is the one
> most collectors get wrong by accident — because a failed run is "not a good example" to a
> naive filter, when in fact **it is the most valuable example in the set**.

### M5 — Native tool-call format and scaffold adaptation

Coder-Next's card explicitly calls out "**adaptability to various scaffold templates**" ✅
— Claude Code, Qwen Code, Qoder, Kilo, Trae, Cline. The `qwen3_coder` tool-call format ✅ is
a documented, machine-parseable wire format that vLLM parses natively ✅.

A model that does not emit a format the harness can parse scores **zero**, regardless of how
good its code is.

> **Consequence:** reuse the base model's chat template and tool-call serialisation **byte for
> byte**, and add a parser-conformance test as a release gate (§5).

---

## 3. Two routes to "3B that behaves like 300B" — and we pick one

### Route A — **Inherit** (MoE, 3B active, large total pool) 🎯 **CHOSEN**

**Base:** `Qwen3-Coder-30B-A3B-Instruct` (30.5B total / 3.3B active, Apache-2.0) ✅
**Teacher:** `Qwen/Qwen3-Coder-Next` (80B / 3B, Apache-2.0) ✅

The 300B-class capability is **already inside the base**. Our contribution is *behavioural
specialisation*:

- the tool protocol our harness uses (M5)
- our recovery style and verification discipline (M3, M4)
- our code conventions and repository idioms
- our honesty policy — report failure as failure

We are not trying to make a 3B model smart. We are making a 3B model **ours**, and we are
adding the behaviours that a generic base model lacks.

| | Route A (Inherit) |
|---|---|
| M1 knowledge | ✅ inherited (30.5B pool) |
| M2 long context | ✅ inherited (262K native) ✅ |
| M3 verified post-training | 🎯 our data + RFT |
| M4 long-horizon | 🎯 our data |
| M5 tool format | ✅ inherited (`qwen3_coder`) |
| Cost | **~$200–900** end to end (§ `09-COST-MODEL.md`) |
| Time | weeks |
| Risk | the fine-tune **damages** what it inherited → caught by gate G4 |

### Route B — **Earn** (dense 3B)

Train a genuine 3B-dense model to 300B-class agentic behavior from scratch-ish.

**This is what VibeThinker-3B did — but for *verifiable reasoning* (math and code contests),
not for agentic tool use** ✅. It took a research team's full effort: curriculum SFT, multi-domain
RL, and offline self-distillation ✅.

No dense 3B has demonstrated 300B-class **agentic** behaviour with tool use. This path needs
RL infrastructure, a verifier, and months.

| | Route B (Earn) |
|---|---|
| M1 knowledge | ❌ must be built — 3B of knowledge is 3B of knowledge |
| M2 long context | ❌ standard attention; much smaller practical context |
| M3 verified post-training | 🎯 our data + RL infra (much more) |
| M4 long-horizon | 🎯 our data |
| M5 tool format | 🎯 our data (no base to inherit from) |
| Cost | **10–100× Route A** 📐 |
| Time | months |
| Risk | high on every axis |

### **Decision: Route A.** 🎯

Because M1, M2 and M5 are inherited rather than earned, our marginal cost to reach
300B-class behavior is **data + a few GPU-hours**, not a research programme. That is the
entire reason this project is viable on a bootstrap budget.

**And the naming follows:** "MenteE Code 3B" = **3B activated parameters**. The model card
states both numbers and the base model identity, because a reader who assumes "3B" means
"3B total" will be misled, and the disclosure is what keeps the claim honest.

---

## 4. What "works like a 300B" means as acceptance criteria

Not a vibe. Each row is a number we publish.

| Capability | 300B-class reference | MenteE Code 3B gate |
|---|---|---|
| Tool-call validity (parse rate with the real harness parser) | 99.9%+ | **≥ 99.5%** — gate **G1** |
| Repo-level task completion | Coder-Next: **70.6** SWE-bench Verified ✅ | 📐 **≥ 30%** SWE-bench Verified, and **≥ base + 3 pts** — gate **G2** |
| Terminal / multi-turn tool use | Coder-Next: **36.2** Terminal-Bench 2.0 ✅ | 📐 ≥ 20% Terminal-Bench 2.0, ≥ base — gate **G2** |
| **Recovery from tool/command error** | the real differentiator | 🎯 **≥ 60%** of injected errors recovered within 3 turns |
| **No invented APIs** | high for frontier-class | 🎯 **≤ 1%** hallucinated imports/symbols on held-out repos |
| **Honest failure reporting** | rare even at 300B | 🎯 **≥ 95%** — claims success only on a green verification |
| Raw code ability (LiveCodeBench / HumanEval+) | VibeThinker-3B: LCBv6 80.2 ✅ | **≤ 2 pts absolute drop** vs base — gate **G4** |
| Cost per successful task | frontier APIs $0.50–3.00 📐 | **≤ $0.10** — gate **G3** |

📐 The SWE-bench Verified target of ≥30% is a *planning estimate*, not a claim. It must be
replaced by a measured number before it appears in any public artifact. Writing an estimate
into a model card is how models get caught.

### 4.1 The narrowness caveat — important, and easy to miss

"300B-class" is scoped to **agentic coding on realistic repositories**. It is *not* 300B-class
for open-domain knowledge, multilingual breadth, or long-tail Q&A. Coder-Next is the strongest
of the three on SWE-bench; it is not the strongest of the three on everything.

**And a fine-tune narrows further.** A student trained only on your trajectory distribution
drifts toward that distribution and away from the base's general competence. This is exactly
why gate **G4** exists, and why the rule "**coverage beats volume**" matters: 20k trajectories
across 200 repos preserve breadth; 20k across 5 repos destroys it.

> 🎯 **The base's 300B-class capability is a loan, not a gift.** Every token of narrow data
> spends some of it. Track the spend with G4 on every training run, not just the first.

---

## 5. Mapping the thesis to work items

This is the part that tells you what to do on the weekend.

| Mechanism | Who provides it | Work item | Status |
|---|---|---|---|
| **M1** sparse knowledge | base model | `02-BASE-MODEL-SELECTION.md` | ✅ decided |
| **M2** cheap long context | base model | serve at 32K, **train at ≥16K** | `06-TRAINING.md` |
| **M3** verified post-training | **us — this weekend** | execution-verified trajectories → SFT → RFT | `05-DATA-COLLECTION-PLAYBOOK.md` |
| **M4** long-horizon recovery | **us — this weekend** | keep the failures and the recovery | `05-DATA-COLLECTION-PLAYBOOK.md` |
| **M5** tool format | base model + our schema | reuse base template; conformance test | `04-DATA-SCHEMA.md`, `07-EVALUATION.md` |

**This is the precise answer to "if we have more data and the best quality data, are we good to go?"**

> **Your data buys M3 and M4. It cannot buy M1, M2, or M5.**
>
> M1, M2 and M5 come from choosing the right base and not breaking it.
> That is the other ~40% — and it is why the base-model decision in
> `02-BASE-MODEL-SELECTION.md` is worth more than the difference between
> 5,000 examples and 50,000.

**Your weekend is therefore the highest-leverage work in the project.** Not because more
data is good in the abstract, but because it is the only thing that supplies the two
mechanisms a base model does not already have.

---

## 6. How we will prove it

Three numbers, in the model card, all measured:

1. **SWE-bench Verified** — MenteE Code 3B vs base vs teacher, identical harness, identical prompts.
2. **Terminal-Bench 2.0** — multi-turn tool use, where a 3B normally falls apart.
3. **Cost per successful task** — the number the beta is actually judged on.

Plus a **regression table** (gate G4) showing what the fine-tune cost us on raw code
ability. Publishing that table is what makes the other three credible.

---

## 7. The one-line version

> A 3B-activated MoE with a large total weight pool, cheap long-context attention,
> execution-verified post-training, and native tool-call format — **behaves like a 300B
> model on agentic coding.**
>
> Three of those five we inherit by choosing the right base. Two — the verified and the
> long-horizon — **only our data can supply.** That is the whole project.

---

## 8. Sources

| Claim | Source | Accessed |
|---|---|---|
| Coder-Next 80B/3B, architecture, 262K ctx, evals 70.6 / 44.3 / 36.2, Apache-2.0, non-thinking, sampling | [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 2026-10-19 |
| Coder-Next "executable task synthesis + RL"; scaffold-template adaptability | Same model card | 2026-10-19 |
| Gated DeltaNet O(1) state + Gated Attention re-anchoring | [n1n.ai architecture analysis](https://explore.n1n.ai/blog/qwen3-coder-next-architecture-performance-analysis-2026-05-22) | 2026-10-19 |
| Laguna XS 2.1: 33B/3B, 256K, SWE-bench Multilingual 63.1, thinking-mode requirement, OpenMDW-1.1 | [Poolside blog](https://poolside.ai/blog/introducing-laguna-xs-2-1) · [model card](https://huggingface.co/poolside/Laguna-XS-2.1) | 2026-10-19 |
| Poolside "async off-policy agent RL" | [Poolside blog](https://poolside.ai/blog/introducing-laguna-xs-2-1) | 2026-10-19 |
| VibeThinker-3B: 3B dense, MIT, LCBv6 80.2, 96.1% LeetCode, Spectrum-to-Signal pipeline (curriculum SFT + multi-domain RL + offline self-distillation) | [model card](https://huggingface.co/WeiboAI/VibeThinker-3B) · [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1) | 2026-10-19 |
| Qwen3-Coder-30B-A3B: 30.5B/3.3B, 128 experts/8 active, 262K, Apache-2.0, sampling params, `qwen3_coder` format | [HuggingFace model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 2026-10-19 |
