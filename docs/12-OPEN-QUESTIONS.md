# Open Questions — what was NOT verified

> The source blueprint's working rules require separating **verified facts, assumptions, and
> recommendations**, and forbid inventing model sizes, licences, provider pricing, GPU
> performance, or benchmark results.
>
> This document is the honest boundary of that separation. Everything listed here is a
> **known unknown**. Nothing in this file is a blocker for the pilot; everything in it is
> something to resolve before a public claim.

---

## 1. Needs verification before the pilot

| # | Question | Why it matters | How to resolve |
|---|---|---|---|
| **V1** | **Exact `transformers` version that supports Qwen3-MoE LoRA** | Our pinned `transformers==4.46.3` ✅ (2024) predates Qwen3-MoE. Picking the wrong version burns a day. | Install the latest stable, run a 10-step load + forward on `Qwen3-Coder-30B-A3B-Instruct`, confirm no import errors. **Do this first.** |
| **V2** | **Whether Unsloth supports our exact base + version pair** | Unsloth pins versions tightly (one report says `transformers==5.5.0` pinned exactly for Qwen3.5 ✅). A mismatch refuses to run. | Try it in the Stage 0 smoke. If it refuses, fall back to plain PEFT + `bitsandbytes` (17.5 GB is Unsloth's number; plain PEFT will be higher — budget for 24 GB and re-measure). |
| **V3** | **Real training throughput (tok/s) for QLoRA on Qwen3-Coder-30B-A3B at 16K** | Every duration and cost in `09-COST-MODEL.md` is 📐 and depends on this. | **Stage 0 measures it.** That is the entire reason Stage 0 exists. |
| **V4** | **Whether `Qwen3-Coder-30B-A3B` still has a `fail_to_pass`-derivable issue set** | Our data collection depends on real GitHub issues with linked failing tests. | Check 10 candidate repos for issue→test linkage before committing to the approach. |
| **V5** | **Teacher generation throughput on a 48 GB card at Q4** | The teacher cost model (§2 of `09-COST-MODEL.md`) assumes ~40 tok/s. | Measure during teacher bring-up (P0.5). 500 tokens of generation tells you the rate. |
| **V6** | **Which OpenCode version the beta targets** | v1 and v2 have different config keys and a known `undefined/chat/completions` bug ✅. | `opencode --version` on day 0; test both config shapes. |
| **V7** | **SWT-bench-style issue-mining tool availability** | Manual issue mining does not scale to 20k. | Evaluate SWE-bench-style mining tooling, or accept synthetic tasks as the primary source. |

---

## 2. Needs verification before the public beta

| # | Question | Why it matters |
|---|---|---|
| **V8** | **Benchmark licences for commercial use** | SWE-bench, Terminal-Bench, LiveCodeBench, HumanEval, BFCL — each has its own licence. A model card that *reports* a benchmark may have obligations attached. Verify per benchmark. |
| **V9** | **Whether Qwen's own chat template supports the `tools=` argument we need** | The card demonstrates tool use ✅, but the exact rendering path (and whether the `tools` schema goes in the template or in the request) must be confirmed from the tokenizer's `apply_chat_template` signature. |
| **V10** | **Second-agent protocol requirements** | Roo Code requires native tool calling ✅; Kilo Code uses `GET /v1/models` ✅. Others may differ. Verify per agent. |
| **V11** | **Current published API prices for the G3 comparison** | Prices move. The comparison must be measured on the day of measurement and cite an access date. |
| **V12** | **Whether the OpenMDW-1.1 + Poolside AUP is acceptable for the fallback base** | A legal question, not an engineering one. Get a real answer before using the fallback — do not assume. |
| **V13** | **Terms of service for a public beta (MenteE AI's own)** | What we promise users, what we log, what we retain. Drafted with the privacy controls in `08-SERVING-AND-BETA.md` §5. |
| **V14** | **The exact per-task cost under real beta concurrency** | The $0.02/task estimate assumes a specific throughput and concurrency. Measure it. |
| **V15** | **Whether vLLM's `qwen3_coder` parser handles every tool-call variant Qwen emits** | Including parallel calls, empty arguments, and unicode in arguments. Build golden cases from the base model's own outputs. |
| **V16** | **Whether the hosted fallback (Together / Fireworks / etc.) would change the cost story** | Only if our own GPU ops become a burden. Revisit at v0.2, not now. |

---

## 3. Deliberately out of scope for v0.1

| Not doing | Revisit |
|---|---|
| A dense-3B MenteE Code | v0.2 — if the MoE route is proven, revisit `VibeThinker-3B`-style dense training (MIT ✅) |
| Reasoning/thinking mode | v0.2 — the teacher is non-thinking ✅, so the student is too |
| Multimodal | v0.3 |
| Multi-agent / parallel tool calls | When the harness supports it |
| A hosted MenteE Code API tier | Post-beta, if the cost model supports a margin |
| Training on the beta's live traffic | Never automatic; always a human-reviewed candidate set |

---

## 4. Assumptions currently baked into the documents

These are stated as 📐 throughout. Each is falsifiable by a measurement we have planned.

| # | Assumption | Where it appears | Resolved by |
|---|---|---|---|
| **A1** | Teacher throughput ≈ 40 output tok/s on a 48 GB card at Q4 | `09-COST-MODEL.md` §2 | P0.5 |
| **A2** | Student throughput ≈ 2,000 output tok/s / 15,000 prefill tok/s at Q4, 24 GB | `09-COST-MODEL.md` §4 | Stage 0 |
| **A3** | Mean trajectory ≈ 8,000 output tokens | `09-COST-MODEL.md` §2 | First 20 trajectories |
| **A4** | Task success rate ≈ 50% | `09-COST-MODEL.md` §4 | G2 |
| **A5** | ~15 model requests per agent task | `09-COST-MODEL.md` §4 | G2 telemetry |
| **A6** | Sandbox verification ≈ 4 min per trajectory | `09-COST-MODEL.md` §3 | P0.2 |
| **A7** | MenteE Code 3B can reach ≥ 30% SWE-bench Verified | `01-THESIS-300B-CLASS.md` §4 | G2 + external eval |
| **A8** | A 3B-activated model saturates around 50k–200k trajectories | `00-EXECUTIVE-SUMMARY.md` §4.1 | Scale experiments |
| **A9** | ≥ 30% of examples can contain a recovery path | `05-DATA-COLLECTION-PLAYBOOK.md` §0 | Collection |
| **A10** | GPU prices stay in the $0.36–$0.71/hr band for 24 GB | `09-COST-MODEL.md` §1 | Re-price at booking |

> 🎯 **Every one of these has a resolution step with an owner and a date.** That is the
> difference between an assumption and a guess.

---

## 5. The one thing I would check first

> 🎯 **V1 — the `transformers` version.** Everything else in the pilot depends on being able
> to load the base model and attach a LoRA adapter. It is a 20-minute check, it is on the
> critical path, and getting it wrong costs a day of confused errors.
>
> `pip install -U transformers` → load `Qwen/Qwen3-Coder-30B-A3B-Instruct` → attach a
> r=16 LoRA → one forward pass. If that works, the rest of Stage 0 is mechanical.

---

*Everything in `00`–`10` was verified against primary sources on 2026-10-19. This file
records the boundary of that verification.*
