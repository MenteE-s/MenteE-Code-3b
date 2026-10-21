# Cost Model

> **Rule from the source blueprint:** *"Produce three estimates... Use current published prices
> when available, record assumptions and access dates, and label estimates separately from
> measured costs. Do not invent exact training durations before profiling a representative
> dataset subset."*
>
> Prices accessed **2026-10-19**. GPU-hour prices are **marketplace medians that move daily** —
> re-check at booking. Training durations are **📐 estimates** until Stage 0 measures real
> throughput.

---

## 1. Verified GPU prices (Vast.ai marketplace)

| GPU | VRAM | On-demand | Spot / interruptible | Note |
|---|---|---|---|---|
| RTX 3090 | 24 GB | $0.22 | $0.09 | Cheapest 24 GB option ✅ |
| **RTX 4090** | **24 GB** | **$0.36 – $0.71** | **$0.14** | 🎯 **The workhorse.** Sources differ (median vs range) |
| RTX 5070 | 12 GB | $0.30 | $0.11 | Too small for the student |
| RTX 5070 Ti | 16 GB | $0.42 | $0.16 | Too small |
| **RTX 5090** | **32 GB** | **$1.18** | **$0.43** | 32 GB fits the student comfortably; your `setup_vast.sh` already targets sm_120 ✅ |
| RTX 5080 | 16 GB | $0.59 | $0.22 | Too small |
| **L40S** | **48 GB** | **$0.47** | **$0.19** | 🎯 **The teacher card.** |
| **RTX A6000** | **48 GB** | **$0.43** | — | 🎯 **The teacher card.** |
| A100 | 80 GB | $0.94 – $1.45 | $0.38 | Overkill for a 3B student; fine for a long Stage-2 run |
| H100 | 80 GB | $0.78 – $2.27 verified; up to $10.84 outlier | $0.78 | ⚠️ Sources disagree wildly — **do not plan on the $10.84 figure** |

**Terms:** per-second billing ✅ · $5 minimum deposit ✅ · storage billed per GB/hour even when stopped ✅ · interruptible 50%+ cheaper ✅.

**Sources:** [Thunder Compute comparison, reviewed 2026-10-16](https://www.thundercompute.com/blog/vast-ai-vs-thunder-compute) ·
[ComputeComparison](https://computecomparison.com/provider/vast-ai) ·
[Spheron](https://www.spheron.network/blog/vastai-pricing-2026) ·
[Vast.ai pricing](https://vast.ai/pricing) ·
[RunPod/Vast/Lambda comparison](https://www.promptquorum.com/local-llms/cloud-gpu-rental-comparison-2026) — all accessed 2026-10-19.

> 🎯 **Planning rate for all estimates below: $0.45/hr for 24 GB and 48 GB cards.** A single
> mid-range number, so the totals are reproducible and not a moving target. Re-price at
> booking.

---

## 2. The cost that actually matters: teacher generation

> **This is the largest line in the project and it is the one the source blueprint omits.**
> It is not in the "teacher API costs" line because **we have no teacher API.** We host the
> teacher.

### 2.1 The arithmetic

| Quantity | Value |
|---|---|
| Trajectories at v0.1 | 20,000 |
| Attempts per trajectory (includes failures we keep) | 1.2 |
| Total attempts | 24,000 |
| Mean output tokens per attempt | 📐 8,000 |
| **Total generated output tokens** | **~192M** |
| Teacher throughput on one 48 GB card, Q4 | 📐 ~40 tok/s (3B active, 80B total, AWQ) |
| GPU-hours on one card | 📐 1,333 |
| GPU-hours on 8 cards in parallel | 📐 167 |
| **Teacher cost at $0.45/hr × 8 cards** | 📐 **~$600** |

**As an API it would be:** 192M output tokens. At Qwen3-Coder-30B-A3B's published
$0.28/M output ✅ that is **~$54**; at a frontier rate of $3–15/M it is **$576–$2,880**.

> 🎯 **The self-hosted teacher is not obviously cheaper at the 30B-A3B API price.** It is
> cheaper at *frontier* prices, and — far more importantly — it removes the licence
> ambiguity about training on a closed provider's outputs, which is the real reason to
> self-host.
>
> If the teacher cost proves painful, the lever is **20k → 5k trajectories**, not switching
> to a closed API. The pilot only needs 500.

### 2.2 The lever that matters most

| Trajectories | Teacher GPU-hours (8 cards) | 📐 Teacher cost |
|---|---|---|
| 500 (pilot) | 42 | **~$19** |
| 2,000 | 167 | ~$75 |
| 20,000 (v0.1) | 1,333 | ~$600 |
| 50,000 (expanded) | 3,333 | ~$1,500 |

> 🎯 **Every stage is gated on the previous one.** The pilot costs ~$19 in teacher time.
> Do not spend the $600 until the $19 experiment has proved the recipe. This is the single
> biggest cost-control lever in the project and it costs nothing to use.

---

## 3. The three estimates

### 3.1 Minimum viable experiment — *prove the loop*

**Scope:** 500 verified trajectories, Stage 0 + Stage 1 training, G1–G4 evaluation. **No public beta.**

| Item | Detail | 📐 Cost |
|---|---|---|
| Teacher generation | 600 attempts × 8K tokens, 1 card | **$19** |
| Sandbox verification | 600 × ~4 min, CPU-only, parallel | **$2** |
| Stage 0 smoke | 24 GB, ~10 min | **$0.10** |
| Stage 1 training | 24 GB, ~3–6 hrs | **$1.50 – $3** |
| Evaluation G1–G4 | 24 GB, ~3 hrs | **$1.50** |
| Storage (datasets, checkpoints) | ~50 GB, a few days | **$1** |
| **Total** | | **$25 – $30** |

**Decision this buys:** whether the fine-tune beats the base model (G2). Everything else is
downstream of that answer.

### 3.2 Recommended initial release — *v0.1 + 10-day beta*

**Scope:** 20,000 verified trajectories, Stage 2 training, Stage 3 RFT, full eval suite, 10-day public beta on a single 24 GB card.

| Item | Detail | 📐 Cost |
|---|---|---|
| Teacher generation | 24,000 attempts, 8 × 48 GB cards, 167 hrs | **$600** |
| Sandbox verification | 24,000 × ~4 min, CPU, parallel | **$30** |
| Stage 0 + Stage 1 | as above | **$5** |
| Stage 2 training (20k) | 4 × 48 GB, ~17 hrs | **$30** |
| Stage 3 RFT | 4,000 generations + training | **$25** |
| Evaluation (G1–G4, E5) | 24 GB + CPU | **$25** |
| **Beta serving** | 240 hrs × 1 × 24 GB card (~$0.50) | **$120** |
| Storage + egress | ~500 GB, 30 days | **$15** |
| Contingency (20%) | | **$170** |
| **Total** | | **~$1,020** |

### 3.3 Expanded experiment — *50k trajectories, 30-day beta, red-teaming*

| Item | 📐 Cost |
|---|---|
| Teacher generation (60,000 attempts, 8 cards) | **$1,500** |
| Sandbox verification | **$100** |
| Stage 2 training (50k) | **$120** |
| Stage 3 RFT + DPO experiments | **$200** |
| Evaluation + red-teaming | **$300** |
| Beta (30 days, 3 × 24 GB) | **$1,080** |
| Storage, egress, monitoring | **$100** |
| Contingency (25%) | **$850** |
| **Total** | **~$4,250** |

> 🎯 **The whole project — minimum to expanded — is under $5,000 in GPU spend.**
> That is the practical answer to "is this affordable". It is affordable *because* the
> teacher is open-weights and the student is 3B-active. A frontier-API teacher would put
> the same project in the $10,000–$50,000 range before a single model was trained.

---

## 4. Cost per successful task (gate G3)

### 4.1 The model

Per agent task, assume 15 model requests:

```
input  ≈ 15 × 40,000  = 600,000 tokens   (mostly cache-hit)
output ≈ 15 ×  4,000  =  60,000 tokens
```

### 4.2 Self-hosted

| Component | 📐 Assumption |
|---|---|
| Aggregate token throughput (Q4, 3B active, 24 GB, concurrency 8) | ~2,000 output tok/s, ~15,000 prefill tok/s |
| Output time | 60,000 / 2,000 = 30 s |
| Prefill time | 600,000 / 15,000 = 40 s |
| **GPU time per task** | **~70 s** |
| GPU rate | $0.50/hr (4090) |
| **Cost per task attempt** | 70/3600 × $0.50 = **$0.010** |
| Task success rate | ~50% |
| **Cost per successful task** | **≈ $0.02** |

### 4.3 The comparison

| Serving option | 📐 $/successful task |
|---|---|
| **MenteE Code 3B, self-hosted on a 4090** | **~$0.02** |
| MenteE Code 3B via a hosted API (at the 30B-A3B published rate ✅ $0.07/M in, $0.28/M out) | ~$0.12 |
| Same base model via a hosted API | ~$0.12 |
| Frontier-class API (~$3/M in, ~$15/M out) | ~$2.70 |

> 🎯 **The headline claim: ~$0.02 vs ~$2.70 — roughly 100× cheaper for a completed task.**
> The exact ratio is an estimate and will be measured (gate G3). The *direction* is not in
> doubt, and the direction is the entire commercial case.

### 4.4 Also report: tokens per successful task

| Metric | 📐 Estimate |
|---|---|
| Tokens per task attempt | ~660,000 (600K in + 60K out) |
| Tokens per successful task | ~1,320,000 |

> 🎯 **Report this number too.** It is provider-independent, it is what users can compare
> across any future provider, and it is the number that survives a pricing change by
> someone else.

### 4.5 The comparison that matters most

| Option | Marginal cost |
|---|---|
| MenteE Code 3B (self-hosted) | **~$0.02 / successful task** |
| **The free public base model** (Qwen3-Coder-30B-A3B, self-hosted) | **~$0.02 / successful task** |
| Frontier API | ~$2.70 / successful task |

> ⚠️ **The honest problem: the base model gives you the same marginal cost.**
>
> The value of MenteE Code 3B is **not** that it is cheaper than the base. It is that it is
> **better on our distribution** (our harness's tool protocol, our recovery behaviour, our
> code conventions, our honesty policy) and that **we own it** — we can improve it with our
> own data, on our own schedule, with no dependency on a third party's release cadence.
>
> This is exactly why **gate G2 (beat your own base) exists**, and why the kill switch
> exists. If MenteE Code 3B does not beat the free base model in the same harness, it has
> no product. 🎯

---

## 5. Non-GPU costs

| Item | 📐 Cost | Note |
|---|---|---|
| HF Hub storage | $0 | Free for public repos; gated for private |
| Domain + hosting | $10–20/mo | A single VM is enough — this is not a scale problem |
| Monitoring | $0 | Prometheus + a free tier, or a spreadsheet |
| Secrets management | $0 | `~/.mentee`-style config, like the sibling repos ✅ |
| Human time | **the real cost** | This is a weekend-scale project, not a team-scale one |

> 🎯 **Deliberately absent:** Kubernetes, a message queue, a managed vector DB, a
> multi-region deployment, a dedicated platform team. None is justified by a measured
> requirement, and each one costs more in attention than in money. The source blueprint
> says to avoid them ✅ and the numbers above confirm the reason.

---

## 6. Cost-control mechanisms

| Mechanism | Effect | Implement |
|---|---|---|
| **Stage gating** | Don't spend the $600 until the $19 pilot passes | `10-ROADMAP-AND-BACKLOG.md` |
| **Interruptible instances** | 50%+ cheaper for training | Vast interruptible ✅; requires resumable training (a Stage-0 criterion) |
| **Per-second billing** | No waste on spin-up/down | Vast default ✅ |
| **Image + dependency caching** | Setup is often 60% of verification wall-clock | Cache per `(repo, base_commit)` |
| **Daily spend cap** | Bounds total exposure regardless of usage | Gateway, `MENTEE_MAX_DAILY_SPEND_USD` |
| **Kill switch** | Stop the bleeding in < 1 min | One env var |
| **Per-key quotas** | One user cannot eat the budget | Gateway |
| **Checkpoint retention (3)** | Rollback without retraining | Training config |
| **Teacher on the same fleet** | No separate infra, no API bill | Already the plan |

---

## 7. What we are NOT paying for

| Not paying for | Why |
|---|---|
| Closed-API teacher tokens | ~$576–$2,880 at frontier rates for 192M output tokens |
| Distributed training / multi-node | The whole model fits on one 24 GB card |
| A managed inference platform | It moves the margin away and complicates the cost story |
| Human annotation at scale | 🎯 **We hand-QC every example instead.** 12 checks × 500 examples ≈ 2 days. It is the highest-value labour in the project. |
| A dedicated infra team | There is no infra. There is one GPU, one gateway, and a script. |

---

## 8. Summary

| Scenario | Total 📐 | Decision it buys |
|---|---|---|
| **Minimum viable** (500 trajectories, no beta) | **$25 – $30** | Does the fine-tune beat the base? |
| **Recommended** (20k, 10-day beta) | **~$1,020** | Ship v0.1 publicly |
| **Expanded** (50k, 30-day beta) | **~$4,250** | v0.2 + red-teaming |

**Cost per successful task: ~$0.02 self-hosted, vs ~$2.70 for a frontier API.** 📐

**All figures are estimates.** The first real measurement is Stage 0's tokens/sec, and the
cost model should be rebuilt from it on day one. 🎯
