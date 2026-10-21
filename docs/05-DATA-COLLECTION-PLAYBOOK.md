# Data Collection Playbook — the Weekend Job

> **You said: "if we have more data, quality best data, then we are good to go?"**
>
> This document is the answer, written as a procedure. It covers exactly what to collect,
> how to collect it, what to throw away, and how to know when to stop.
>
> It assumes you have: a rented GPU (48 GB for the teacher), Docker, and the harness
> (`MenteE SWE` from the sibling repo, or OpenCode headless).

---

## 0. The one idea that determines the value of your weekend

Your data buys exactly two of the five mechanisms that make a 3B model behave like a 300B
model (`01-THESIS-300B-CLASS.md`):

- **M3 — execution-verified post-training**
- **M4 — long-horizon recovery**

The other three (**M1** sparse knowledge, **M2** cheap long context, **M5** native tool-call
format) come from the base model and are inherited for free.

**So the question is not "how many examples?" It is "how many of my examples are
*verified* and *contain a recovery*?"**

> 🎯 **A 500-example dataset where 100% is verified and 30% contains a failure→recovery path
> is worth more than a 5,000-example dataset that is 40% verified and 2% has recovery.**
> This is not a stylistic preference. It is the difference between a model that completes
> tasks and a model that emits plausible code.

### 0.1 The three numbers to track while you collect

Keep a running count. These are the only three numbers that matter on day one:

| Metric | Target (v0.1) | Why |
|---|---|---|
| **Verified rate** | **≥ 80%** of collected examples | Unverified data teaches imitation, not completion |
| **Recovery rate** | **≥ 30%** of examples contain a failed-then-recovered step | This is M4. This is what makes it an *agent* |
| **Repo diversity** | **≥ 1 example per repo, ≥ 50 repos for the first 500** | Coverage beats volume. 5 repos is not a dataset |

---

## 1. Choosing what to collect

### 1.1 The task distribution you actually want

| Task type | Share | Why |
|---|---|---|
| `bug_fix` | 40% | The single most common real agent task, and the one with a clean verifier |
| `feature` | 20% | Tests the model's ability to write *new* code + tests |
| `refactor` | 15% | Multi-file, requires understanding structure before editing |
| `test_authoring` | 10% | High value: teaches the model to write its own verification |
| `build_fix` / `dependency_upgrade` | 10% | Realistic messiness — env problems, not logic problems |
| `perf` | 5% | Nice to have, noisy verifier, later |

### 1.2 Repo selection rules

**Licence filter (hard):** Apache-2.0, MIT, BSD-2/3, ISC. GPL-3.0 and AGPL-3.0 are
*usable for training* but carry copyleft obligations on distribution — get a legal opinion
before shipping a model trained on them. **Research-only / "non-commercial" / unspecified →
excluded.** 🎯

**Size band:** the sweet spot is repos of **1k–60k lines** that build and test in under 5
minutes.

| Band | Verdict |
|---|---|
| < 500 lines | Trivial. Teaches nothing about navigation. |
| 500 – 1k | Too small; usually a single-file library. |
| **1k – 60k** | ✅ **Target band.** Real structure, fast test loop. |
| 60k – 300k | Acceptable. Slow setup; include some. |
| > 300k | ❌ Setup alone can exceed the timeout. Rarely worth it. |

**Must have:**
- ✅ A test suite that runs in < 5 min and has **at least 10 test functions** (so `pass_to_pass` is meaningful)
- ✅ A `setup` that works offline-ish or with a pinned requirements file
- ✅ A public commit history (so `base_commit` is real and `fail_to_pass` can be derived)
- ✅ Not a benchmark repo (see §1.4)

**Should have:** a linter/typechecker config (teaches the model to run the linter), a CI config, multiple languages.

### 1.3 Language mix

Start **Python-dominant** (60–70%) because the verifier ecosystem is best there, then add:

| Language | Share | Notes |
|---|---|---|
| Python | 60% | Primary. Best verifiers, best tooling. |
| TypeScript / JavaScript | 20% | `npm test` / vitest. Growing agent usage. |
| Go | 10% | `go test ./...` is fast and hermetic — excellent verifier. |
| Rust | 5% | Compile time is the bottleneck, but the verifier is perfect. |
| Java / Kotlin | 5% | Slow builds; include a few only. |

> 🎯 **Do not add a language until you have 500 verified examples in your current mix.**
> Every new language costs setup debugging, and a broken setup silently produces
> "verification failed" examples that get filtered out — you will think you collected data
> and collected nothing.

### 1.4 Contamination — the exclusion list

**These repos are BANNED from training data** (they are our evaluation set):

- SWE-bench / SWE-bench Verified / SWE-bench Lite / SWE-bench Pro / SWE-bench Multimodal
- Terminal-Bench
- LiveCodeBench
- HumanEval / MBPP / EvalPlus
- SWE-Gym / R2E-Gym / SWE-smith instance sets
- Any repo in `configs/dataset_config.json` → `excluded_repos`

**Enforce it in the registry, not in a filter.** Mark the source `role: "eval_only"` and the
training pipeline cannot see it. 🎯

### 1.5 Where the tasks come from

Three sources, in order of preference:

| Source | How | Verifier quality | Notes |
|---|---|---|---|
| **Real GitHub issues** ✅ | Scrape issues with a linked failing test | ⭐⭐⭐ best | Realistic, and `fail_to_pass` is derivable from the issue's fix commit |
| **Synthetic issues from a teacher** | Teacher reads a repo and writes a task + a failing test | ⭐⭐ good | Cheapest at scale; must be sanity-checked by a human |
| **Hand-written tasks** | You write them | ⭐⭐⭐ best | Use for the seed set; too slow for volume |

> 🎯 **Recommended split for the first 500: 200 real issues + 250 synthetic + 50
> hand-written.** The hand-written 50 are your quality canary — you personally review every
> one of them, and they set the standard for the rest.

---

## 2. How to collect — the loop

```
  ┌──────────────────────────────────────────────────────────┐
  │  FOR EACH TASK:                                           │
  │                                                          │
  │  1. clone repo @ base_commit   (network ON — collection) │
  │  2. build sandbox image, install deps (cache aggressively)│
  │  3. run test suite → BASELINE  (must be green, else skip)│
  │  4. apply the reference patch → EXPECTED (must be green)  │
  │  5. revert patch → must reproduce the FAILING test        │
  │  6. NOW: let the TEACHER attempt the task from scratch   │
  │  7. record EVERY turn, including failures                │
  │  8. re-run the test suite on the teacher's patch         │
  │  9. compute fail_to_pass / pass_to_pass                  │
  │ 10. if it failed → KEEP IT, tag as failed_but_honest     │
  └──────────────────────────────────────────────────────────┘
```

### 2.1 Steps 1–5 are the part everyone skips, and it is the whole point

**You must prove the task is real before you let the teacher attempt it.**

- Step 3 (baseline green) → the repo is healthy; you are starting from a working state.
- Step 4 (patch → green) → the reference fix works; the task is solvable.
- Step 5 (revert → red) → the failing test genuinely detects the bug.

If you skip step 5, you will collect tasks whose "fix" is unmeasurable, and your
`fail_to_pass` will be empty — which means your verifier proves nothing. **A task with an
empty `fail_to_pass` must be rejected, not kept with a warning.** 🎯

**Cost-saving tip:** steps 1–5 are teacher-free and cacheable. The expensive part is step 6
(teacher generation). Validate the task, *then* pay for the attempt. Also cache the built
Docker image per `(repo, base_commit)` — setup is often 80% of the wall-clock.

### 2.2 Step 6 — the teacher run

**Teacher:** `Qwen/Qwen3-Coder-Next` (80B/3B active, Apache-2.0 ✅) on a 48 GB card.
**Sampling from the model card:** `temperature=1.0, top_p=0.95, top_k=40` ✅.

**Use a real agent harness**, not a bare completion loop — otherwise you are collecting
text generation, not agent behaviour. Either:

- **MenteE SWE** (sibling repo) — you already own it, and tuning the model to your own
  harness is a strategic asset, and
- **OpenCode** headless — the actual target harness for the beta.

> 🎯 **Use the same harness you will serve into.** Training on MenteE SWE's tool protocol and
> then serving into OpenCode means the model has to learn a second tool dialect for no
> reason. If the beta's target is OpenCode, generate with OpenCode.
>
> If you do use both, **tag every trajectory with `generator.harness`** so you can slice the
> dataset by dialect later and measure which one teaches what.

**Turn budget:** 20–40 turns per task. Cap it. A run that goes to 150 turns is a rabbit hole
and teaches wandering.

**Sampling for diversity:** vary `temperature` across 0.7–1.0 and vary the *task
phrasing* across attempts. Ten attempts at the same task with identical settings give you
ten near-duplicates — the deduper will collapse them and you will have wasted the compute.

### 2.3 Step 7 — record everything, especially the failures

**Do not filter during collection.** Collect first, filter later with recorded reasons. A
filter applied at collection time is a filter you cannot audit, and you will not know what
you lost.

Specifically, keep:
- turns where a tool call **errored**
- turns where the agent **read the wrong file first**
- turns where the agent **proposed a wrong hypothesis** and then corrected itself
- the **final answer even when the task failed**

### 2.4 Step 8–9 — verification

Re-run the test suite in the sandbox against the teacher's patch:

- `fail_to_pass` tests: FAIL before, PASS after → the task is **solved**
- `pass_to_pass` tests: must all still PASS → no regression
- any `pass_to_pass` regression → **reject the example** (teaches breakage)

`verification.status` is written by the sandbox. Never by the model. Ever. 🎯

### 2.5 Step 10 — the failure branch

**If the teacher failed the task, keep the trajectory** and set:

```jsonc
"task": { "outcome": "failed_but_honest" }
"verification": { "status": "fail" }
"quality": { "verified": false, "checks": { "has_failure_path": true } }
```

These are among the most valuable examples in the set. They teach:
- the model to keep going after a failed attempt (a real recovery path)
- the model to report failure honestly instead of claiming success

> 🎯 **A dataset with a 100% success rate has thrown away its most valuable examples.**
> Target: 20–40% of examples are `failed_but_honest` or contain a mid-run recovery.

---

## 3. Manual quality control — the 12-point gate

Every example is checked. It is fast (a human reads 60–100/hour) and it is the difference
between a dataset and a pile of JSON.

| # | Check | Fail if |
|---|---|---|
| 1 | **Task is well-posed** | The instruction is ambiguous, or requires info not in the repo |
| 2 | **`fail_to_pass` non-empty** | The task has no detectable failing test |
| 3 | **Task is non-trivial** | Fixable by changing one line with no understanding required |
| 4 | **The agent read before it wrote** | It patched a file it never opened |
| 5 | **Paths are real** | Any edited/created path does not exist in the repo |
| 6 | **No hallucinated APIs** | The patch calls a function that does not exist anywhere |
| 7 | **The patch applies** | Re-applying the diff to the base commit fails |
| 8 | **No regressions** | Any `pass_to_pass` test broke |
| 9 | **No truncation** | Any tool output was shortened (teaches hallucination) |
| 10 | **No secrets** | No API keys, tokens, `.env` contents, credentials |
| 11 | **Turns are sensible** | No infinite loops, no repeated identical tool calls, no thrash |
| 12 | **Honest conclusion** | It claims success but the verification says `fail` |

**Any fail → reject, with a reason code.** Rejected examples are stored, not deleted.

### 3.1 The four checks people skip and shouldn't

**#4 (read before write).** A trajectory where the agent patches a file it never opened
teaches the model to guess. In a 3B model, guessing produces confident garbage.

**#6 (hallucinated APIs).** This is the single most common quality problem in agentic
datasets, and it is **directly relevant to your 300B-class goal**: a model that invents
APIs looks exactly like a small model, no matter how good the base is. Gate it hard.

**#9 (truncation).** If a tool output was cut short, the model never saw the rest — and
training on it teaches the model to *invent* the rest. This is poison, and it is invisible
unless you check for it.

**#12 (honest conclusion).** The model says "fixed, all tests pass" but the sandbox says
`fail`. **Reject the example.** Keeping it teaches the exact behaviour that breaks the
product's core promise.

---

## 4. The sandbox (do not skip this)

The verifier is what makes the data worth anything. It is also the biggest engineering
task in Phase 1 — budget **2–3 days** for it, not 2 hours.

### 4.1 Requirements

| Requirement | Why |
|---|---|
| **Disposable** — new container per run | One test that corrupts state poisons the next |
| **No network** | A test that reaches the internet is non-deterministic and can exfiltrate |
| **CPU / memory / time caps** | A hung test must not burn the budget |
| **Read-only mount of the repo except the workdir** | A test cannot rewrite the source tree |
| **No host secrets** | Never mount `$HOME`, never pass the API key into the container |
| **Deterministic seeds** | `PYTHONHASHSEED=0`, `TZ=UTC`, fixed `pytest -p no:randomly` |

### 4.2 Practical implementation

```bash
# inside the container, as the runner
cd /workdir
git apply --check /patches/candidate.patch || exit 2   # patch must apply
git apply /patches/candidate.patch
timeout 600 python -m pytest -q -p no:randomly tests/ 2>&1 | tail -200
```

Exit code 0 = all passed. The runner parses per-test results to build `fail_to_pass` and
`pass_to_pass`.

> ⚠️ **The single biggest practical failure mode:** the container cannot install
> dependencies, so every test "fails" for the wrong reason, and you collect a dataset of
> garbage that looks fine. 🎯 **Prove the sandbox on 5 known-good repos before you collect
> anything.** If the baseline is not green on a repo you know is healthy, the sandbox is
> broken, and every example you collect is worthless.

### 4.3 Cost control

- **Cache the built image per `(repo, base_commit)`** — setup dominates wall-clock.
- **Parallelise** across repos — the verifier is CPU-bound, not GPU-bound. A cheap 16-core
  CPU box can run 20–50 sandboxes concurrently.
- **Skip the teacher entirely for repos whose sandbox fails** — never pay GPU time on a task
  you cannot verify.

---

## 5. Filtering and dedup

### 5.1 Dedup (before QC, after normalisation)

| Method | Catches | Threshold |
|---|---|---|
| Exact sha256 of normalised content | Literal duplicates | — |
| MinHash / SimHash on the instruction + patch | Near-duplicates from repeated attempts | 🎯 0.9 cosine |
| Repo + base_commit + task signature | Same task collected twice | exact |

> 🎯 **Keep the best instance of a near-duplicate cluster, not the first.** "Best" =
> highest QC score, then fewest turns, then most recent.

### 5.2 Filters

**Hard rejects:**
- schema invalid
- `fail_to_pass` empty
- any `pass_to_pass` regression
- secret detected
- hallucinated path or API
- truncated tool output present
- from a benchmark repo
- from a non-trainable source

**Soft scores (recorded, not rejected):** turn count, patch size, task difficulty, presence
of a recovery path.

---

## 6. Sizing — when to stop

### 6.1 The stop signal

> 🎯 **Stop collecting when the 500-example LoRA experiment beats the base model on the
> in-harness suite (gate G2), and the marginal example is no longer improving it.**

That is the correct stopping rule, and it means the weekend collection should be
**batch-oriented, not exhaustive**: produce a clean 500 first, then keep producing batches
while the training loop iterates.

### 6.2 Expected scale bands

| Volume | What it buys | Cost 📐 |
|---|---|---|
| 100 verified | Smoke test. Validates the pipeline end to end. | ~$10 |
| **500 verified** | 🎯 **The experiment that decides the project.** | **~$30** |
| 2,000 verified | First real training run. | ~$100 |
| 20,000 verified | v0.1 release scale. | ~$400–900 |
| 50,000+ | 📐 Diminishing returns for a 3B. Diversify or move to RL instead. | ~$1,500+ |

**Above ~50k, the question stops being "how much" and becomes "how diverse".** More
trajectories from the same 20 repos will make the model *worse* at the real world by
narrowing it. 🎯

### 6.3 Where to spend the marginal hour

Ranked by expected value per hour of your weekend:

1. **More repos** (breadth) — beats more examples per repo, every time.
2. **More `failed_but_honest` / recovery examples** — this is M4, and it is what separates
   an agent from a code generator.
3. **More verified examples** (raise the verified rate from 60% → 90%) — directly worth more
   than raw volume.
4. **More languages** — only after 1–3 are solid.
5. **More examples per repo** — the *last* thing to optimise, and the first thing to stop.

---

## 7. Your weekend checklist

### Saturday

- [ ] **Set up the repo** — git init, `src/` skeleton from `03-ARCHITECTURE.md` §4
- [ ] **Write the source registry** with your first 30 candidate repos; **verify each licence by reading it** (not from memory)
- [ ] **Build the sandbox** and prove it on 5 known-good repos (baseline green)
- [ ] **Bring up the teacher** — vLLM + `Qwen3-Coder-Next` on a 48 GB card
- [ ] **Wire the harness** — OpenCode headless, writing to the schema
- [ ] **Collect 20 end-to-end** and hand-QC all 20. Fix every failure you find.
- [ ] **Check the failures**: are the "failures" real failures, or sandbox problems? (This is where you lose a weekend if you're not careful.)

### Sunday

- [ ] **Scale collection** — run until you have **500 verified**
- [ ] **QC every example** against the 12-point gate
- [ ] **Compute the three numbers**: verified rate, recovery rate, repo count
- [ ] **Dedup + filter**, with recorded reasons
- [ ] **Split** by commit date and repo group
- [ ] **Publish `dataset-v1.0.0`**: JSONL + manifest + quality report
- [ ] **Hand over to the training pipeline** — it should run on your data with no edits

### Sunday evening — the gate

```
verified_rate  >= 80%   ?  →  else: fix the pipeline, not the data
recovery_rate  >= 30%   ?  →  else: collect more failures, they are free
repo_count     >= 50    ?  →  else: diversify, do not add volume
schema_valid   100%     ?  →  else: fix the validator, do not hand-fix records
```

**If all four pass, the weekend succeeded and the 500-example experiment can start
immediately.** 🎯

---

## 8. What to do when something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Every verification fails | Sandbox cannot install deps | Fix the image. Do **not** collect until baseline is green |
| All examples look identical | Same task, same seed | Vary temperature **and** task phrasing |
| Very few `has_failure_path` | You filtered failures out | Stop filtering during collection |
| Duplicate rate > 30% | One task collected many times | Vary tasks, raise dedup threshold |
| Model-invented APIs in the data | Teacher is weak at this repo | Add a read-before-write check (QC #4) and filter |
| The model produced short trajectories | Harness cut off early | Check the turn cap and the tool-output limits |
| `pass_to_pass` regressions everywhere | Teacher is patching too aggressively | Keep only examples with zero regressions; add QC #8 |

---

## 9. The honest closing note

You asked whether more data + the best quality data is enough. It is **most** of it — but
the parts it is *not* are the parts that are cheap to get right and expensive to discover
late:

- the base-model choice (already made, in `02-BASE-MODEL-SELECTION.md`)
- the tool-call format conformance (a schema + template rule, in `04-DATA-SCHEMA.md`)
- the eval discipline (in `07-EVALUATION.md`)

Get those three right and your weekend's data is the difference between a product and a
demo. Get them wrong and no amount of data saves it. 🎯

---

*All licence, model and pricing facts verified against primary sources on 2026-10-19. See
`12-OPEN-QUESTIONS.md` for what remains unverified.*
