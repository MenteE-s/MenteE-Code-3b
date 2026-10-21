# Roadmap & Prioritized Backlog

> Planning targets, not guarantees. The source blueprint's aggressive timeline is kept, with
> the blockers and the critical path made explicit.
> 📐 = estimate · ✅ = verifiable acceptance criterion

---

## 1. The shape of the plan

```
WEEKEND (you)  ──▶  500 verified trajectories  ──▶  dataset-v1.0.0
                                                          │
DAY 0-1  ──▶  connect OpenCode to a local vLLM  ─────────┤
DAY 0-2  ──▶  sandbox verifier  ─────────────────────────┤
DAY 2-3  ──▶  Stage 0 smoke + Stage 1 pilot  ────────────┤
                                                          ▼
DAY 3    ──▶  ★ GATE  ★  does it beat the base?  ──▶  yes ──▶  scale to 20k
                                                      │
                                                      └──▶  no ──▶  KILL SWITCH
                                                                   (§ `00-EXECUTIVE-SUMMARY.md` §3.1)
```

> 🎯 **The only irreversible spend is the data.** The pilot costs ~$30. The release costs
> ~$1,020. Do not spend the second number until the first one has answered the question.

---

## 2. Day-by-day

### Days 0–1 — Infrastructure (no data yet)

| # | Task | Owner | Parallel? | Blocker? |
|---|---|---|---|---|
| 0.1 | `git init`, repo skeleton from `03-ARCHITECTURE.md` §4 | | | → everything |
| 0.2 | **Connect OpenCode to a local vLLM, complete one trivial task** | | | 🎯 **highest-value 20 minutes in the project** |
| 0.3 | Source registry with 30 candidate repos; **verify each licence by reading it** | you | ✅ | → 0.6 |
| 0.4 | Base model download + a 5-minute QLoRA smoke on 20 examples | | ✅ | → 0.2 |
| 0.5 | **Sandbox: prove it on 5 known-good repos (baseline green)** | | ✅ | → collection |
| 0.6 | Teacher serving (vLLM + Coder-Next on 48 GB) | | ✅ | → collection |

> 🎯 **0.2 and 0.5 are the two that silently block everything.** If OpenCode can't talk to
> vLLM, every downstream measurement is wrong. If the sandbox can't produce a green
> baseline, every collected example is garbage. Both are cheap to verify and catastrophic
> to skip.

### Days 2–3 — Data + the pilot

| # | Task | Depends on | Cost 📐 |
|---|---|---|---|
| 1.1 | Collect 20 trajectories end-to-end; hand-QC all 20 | 0.3, 0.5, 0.6 | $3 |
| 1.2 | **Fix every failure found in 1.1** | 1.1 | — |
| 1.3 | Scale collection to **500 verified** | 1.2 | $19 |
| 1.4 | Hand-QC all 500 (12-point gate) | 1.3 | your time |
| 1.5 | Dedup + filter with recorded reasons | 1.4 | $1 |
| 1.6 | Split by commit date + repo group; publish `dataset-v1.0.0` | 1.5 | $1 |
| 1.7 | Stage 0 smoke (20 steps, resumable) | 0.4 | $0.10 |
| 1.8 | **Stage 1 pilot training** (500 ex, 16K seq, 2 epochs) | 1.6, 1.7 | $3 |
| 1.9 | G1 + G2 + G4 evaluation, base vs student, identical prompts | 1.8 | $2 |

### Day 3 — ★ THE GATE ★

```
G2: student task_completion_rate >= base task_completion_rate  (same 50 tasks, same prompts)
G4: LiveCodeBench / HumanEval+ drop <= 2 points absolute
G1: tool-call parse rate >= 99.5% at serving quantisation
```

- **All pass** → scale collection to 2k, then 20k. Start Stage 2.
- **G2 fails** → one iteration of: different LoRA config (r=32 all-linear) / data mix change /
  different base. **Two iterations max.**
- **Still failing** → 🎯 **kill switch.** Ship the base model with the improved harness, or
  re-scope to a serving product. Decide in writing, on day 3, not week 6.

### Days 4–5 — Scale + release candidate

| # | Task | Cost 📐 |
|---|---|---|
| 2.1 | Collection to 20k verified | $600 |
| 2.2 | Stage 2 training (4 × 48 GB) | $30 |
| 2.3 | Stage 3 RFT (500 tasks × 8 attempts) | $25 |
| 2.4 | Merge + AWQ quantise; validate the artefact | $2 |
| 2.5 | G1–G4 + E5 on the **quantised** model | $25 |
| 2.6 | Model card + dataset card, generated from eval JSON | — |

### Days 5–7 — Integration + security + release decision

| # | Task |
|---|---|
| 3.1 | Gateway: auth, quotas, usage accounting, spend cap, kill switch |
| 3.2 | OpenCode integration verified end to end against the **deployed** endpoint |
| 3.3 | Second agent (MenteE SWE or Kilo Code) verified |
| 3.4 | E5 safety suite green; sandbox escape tests pass |
| 3.5 | Privacy: consent flow, retention, deletion, redaction, opt-out — **all tested** |
| 3.6 | **Kill-switch drill** (flip it, confirm 503, flip back) |
| 3.7 | **Launch gate review** — all gates green? |

### Days 8–17 — 10-day public beta

Per `08-SERVING-AND-BETA.md` §4.3.

---

## 3. Parallelisation

| Can run in parallel | Cannot |
|---|---|
| Sandbox build ←→ teacher bring-up | Collection ←→ sandbox (must be proven first) |
| Registry/licensing ←→ anything | Pilot training ←→ dataset publish |
| Stage 0 smoke ←→ collection | Release decision ←→ G2 |
| Eval harness build ←→ training | |
| Model card drafting ←→ final numbers | |
| **Data collection ←→ everything else** | |

> 🎯 **Data collection is the long pole and it is almost entirely parallelisable with
> everything else.** The only hard constraint is that the *sandbox must be proven before
> collection starts*. That is why 0.5 is on Day 0.

---

## 4. Prioritized backlog

### P0 — blocking; nothing else matters until these are done

| ID | Task | Acceptance criteria ✅ | Est |
|---|---|---|---|
| **P0.1** | OpenCode ↔ vLLM connectivity | A trivial task completes in OpenCode against a local vLLM, using the `qwen3_coder` tool-call parser. Both v1 and v2 config shapes tested. | 20 min |
| **P0.2** | Sandbox verifier | On 5 known-good repos: baseline green, reference patch → green, reverted patch → red. Sandbox produces a per-test JSON verdict. | 1–2 days |
| **P0.3** | Source registry | 30 candidate repos, each with a **read** licence, `trainable`, `redistributable`, `role`. All benchmark repos marked `eval_only`. | 3 hrs |
| **P0.4** | Canonical schema + validator | `schemas/mentee-trajectory-v1.schema.json` exists; validator exits non-zero on a bad record; the example validates. | 4 hrs |
| **P0.5** | Teacher serving | vLLM serves Coder-Next; 500 tool-call turns generate; throughput measured (tok/s). | 1 hr |
| **P0.6** | 20 end-to-end trajectories | 20 collected, hand-QC'd, and **every failure fixed**. | 1 day |
| **P0.7** | Stage 0 smoke | 20 steps, finite loss, checkpoint saves + reloads, **resumes from step 15**, and produces a parseable tool call. | 1 hr |
| **P0.8** | 500 verified + QC | 500 examples, verified rate ≥ 80%, recovery rate ≥ 30%, ≥ 50 repos, 100% schema-valid. | your weekend |
| **P0.9** | Pilot training + G2/G4 | Student ≥ base on the 50-task suite; ≤ 2 pts drop on G4; G1 ≥ 99.5% at Q4. | 1 day |
| **P0.10** | Kill switch | `MENTEE_SERVICE_ENABLED=false` → 503 within 5 s. Tested, not assumed. | 30 min |

### P1 — required for a public beta

| ID | Task | Acceptance criteria ✅ | Est |
|---|---|---|---|
| **P1.1** | 20k collection | 20,000 verified, coverage across ≥ 150 repos and 4 languages. | 3 days |
| **P1.2** | Stage 2 + RFT | RFT improves G2 over SFT-only. | 1 day |
| **P1.3** | Gateway | Auth, per-key quota, spend cap, usage JSON, kill switch. | 1 day |
| **P1.4** | Second agent | A non-OpenCode agent completes a task against the deployed endpoint. | 2 hrs |
| **P1.5** | E5 safety suite | 120 cases pass, including 25 prompt-injection cases. | 1 day |
| **P1.6** | Privacy controls | Consent off by default; redaction runs on stored data; deletion removes raw + curated + derived; opt-out is immediate. **All tested end to end.** | 1 day |
| **P1.7** | Model card + dataset card | Generated from eval JSON. Every number measured. Base model, licence and attribution stated. | 4 hrs |
| **P1.8** | Second-agent protocol | Confirmed which agents use `/v1/chat/completions` vs `/v1/responses` ✅, and each is handled. | 2 hrs |

### P2 — after the beta

| ID | Task |
|---|---|
| P2.1 | DPO experiment (only if RFT has plateaued) |
| P2.2 | SGLang serving fallback (≥ 0.5.8 for Coder-Next ✅) |
| P2.3 | Multi-GPU teacher scaling |
| P2.4 | `mentee-embeddings` integration for repo-level semantic retrieval |
| P2.5 | Longer context (64K+) with the train/serve gap closed |
| P2.6 | 30-day beta + paid tier |
| P2.7 | Consider Qwen3-Coder-Next as the base (stretch) |

---

## 5. Critical path

```
P0.2 sandbox ──▶ P0.6 20 trajectories ──▶ P0.8 500 verified ──▶ P0.9 pilot ──▶ ★ GATE ★
                                              ▲                                          │
                                              │                                    yes ───┘
                                       (your weekend)
```

**Everything else is off the critical path.** In particular, the gateway, the model card, the
beta plan, and the second agent can all be built while the data is being collected. 🎯

> 🎯 **The critical path is: sandbox → 20 → 500 → pilot → gate.** Four steps. If you
> protect nothing else, protect these.

---

## 6. Blocker register

| Blocker | Blocks | Mitigation |
|---|---|---|
| Sandbox cannot install dependencies | All collection | Prove on 5 repos before collecting. Cache images. |
| Tool-call format mismatch | G1, everything | Reuse the base chat template; conformance test from Day 0 |
| No GPU available / Vast out of credit | Training | Keep a second provider (RunPod ✅, Lambda) configured |
| Teacher throughput too slow | Collection | Parallel cards; reduce trajectories; reduce per-attempt tokens |
| Dataset licences turn out restrictive | Collection | Registry gate **before** collecting, not after |
| Training OOM at 16K | Stage 1 | `use_gradient_checkpointing: true`; drop to 8K **and** serve at 8K (do not mismatch) |
| QLoRA import breaks on the MoE | Stage 1 | Let the loader convert from 16-bit ✅; try `experts4bit-qlora` (7.16 GB ✅) |
| Eval suite not built when the pilot finishes | Gate | Build G1 early — it is fast and independent |
| OScope creep: scope creep | Timeline | Any new feature needs a gate-deleting argument |

---

## 7. Definition of done — v0.1

- [ ] Model published on HF with a complete, generated model card
- [ ] Dataset card published with the quality report and licence manifest
- [ ] G1 ≥ 99.5% at serving quantisation
- [ ] G2 ≥ base on the 50-task suite, with a Wilson interval reported
- [ ] G3 ≤ $0.10 per successful task, measured
- [ ] G4 ≤ 2 pts drop on LiveCodeBench / HumanEval+
- [ ] E5 green, including prompt injection
- [ ] OpenCode works against the public endpoint, and one other agent
- [ ] Privacy controls live and **tested**
- [ ] Kill switch tested
- [ ] Rollback to the previous checkpoint proven
- [ ] Every published number traceable to a JSON artefact
- [ ] Limitations documented honestly

---

## 8. What we deliberately do not do before the beta

| Deferred | Why |
|---|---|
| Plan mode / todos / routing (from the MenteE SWE roadmap) | They are runtime features, not model features. The beta needs the model. |
| DPO | RFT first — we have a verifier, so labels are free |
| Multi-node training | Fits on one card |
| A managed vector DB for memory | Overkill for a 3B's context window |
| A second base model in parallel | One base, done well. Adding a second doubles the evaluation burden for no v0.1 benefit. |
