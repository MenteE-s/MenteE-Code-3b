# Canonical Data Schema & Processing Workflow

> Machine-readable schema: [`schemas/mentee-trajectory-v1.schema.json`](../schemas/mentee-trajectory-v1.schema.json)
> Example record: [`schemas/examples/example-trajectory.json`](../schemas/examples/example-trajectory.json)
> Canonical interchange format: **JSONL** (one trajectory per line). Parquet is an optional
> analytical export, never the source of truth.

---

## 1. Design principle

> **A trajectory is not a conversation. It is a claim that a task was completed, plus the
> evidence for that claim.**

The schema therefore separates three things that most datasets conflate:

| Block | What it holds | Why separate |
|---|---|---|
| `task` | what was asked, from what starting state | Reusable across teachers, models and harnesses |
| `trajectory` | what the agent did — the turns, the tool calls, the outputs | The learning signal |
| `verification` | **independent ground truth** about what happened | The teacher model saying "done" is not evidence. Only the verifier is. |

**This separation is the single most important structural decision in the schema.** It is what
makes the dataset re-usable with a different teacher, re-verifiable later, and safe against a
model that learns to *claim* success rather than *achieve* it.

---

## 2. Top-level record

```jsonc
{
  "schema_version": "1.0.0",
  "id": "traj_2026-10-19_000123",
  "provenance": { ... },     // where did this come from, under what licence
  "task":      { ... },      // what was being asked
  "environment": { ... },    // the repo + state it happened in
  "trajectory": { ... },     // the turns
  "verification": { ... },   // ground truth
  "quality":   { ... },      // computed by the pipeline
  "split":     { ... }       // train / val / test + why
}
```

### 2.1 `provenance` — required, non-negotiable

```jsonc
"provenance": {
  "source_id": "mentee_synth_v1",      // → registry entry
  "source_url": "internal",
  "source_sha256": "…",                // pin the exact artefact
  "license": "Apache-2.0",             // SPDX id of the *data*, not the weights
  "license_verified": true,            // a human/automated check ran
  "redistributable": true,
  "trainable": true,                   // explicitly "may we train on this?"
  "teacher": {
    "model": "Qwen/Qwen3-Coder-Next",
    "revision": "main",                // pin it; `main` moves
    "license": "Apache-2.0",
    "sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 40 },   // from the model card ✅
    "quantization": "awq",
    "served_by": "self-hosted"
  },
  "generator": { "code_version": "git:abc1234", "prompt_template": "swe_agent_v3" },
  "collected_at": "2026-10-19T14:03:11Z",
  "collected_by": "human:syab"
}
```

> 🎯 **`trainable` is a separate boolean from `license`.** Many licences permit *use* but
> restrict *training on*. These are different permissions and conflating them is how a
> project ships a dataset it had no right to train on. If `trainable` is not `true`, the
> licence gate rejects the example.

### 2.2 `task`

```jsonc
"task": {
  "instruction": "Fix the off-by-one in parse_window() when the window is empty.",
  "issue_url": "https://github.com/org/repo/issues/412",
  "language": "python",
  "task_type": "bug_fix",        // bug_fix | feature | refactor | test_authoring | doc | perf
  "difficulty": "medium",        // easy | medium | hard  (human label)
  "constraints": ["Do not change the public signature of parse_window"],
  "acceptance_criteria": [
    "pytest tests/test_parser.py::test_empty_window passes",
    "no new public API"
  ]
}
```

### 2.3 `environment` — the part that makes re-verification possible

```jsonc
"environment": {
  "repo": "org/repo",
  "repo_license": "MIT",
  "base_commit": "9f2c1ab…",          // exact commit — the repo moves
  "setup_script": "python -m pip install -e . && pip install -r requirements-dev.txt",
  "test_command": "python -m pytest -q tests/",
  "fail_to_pass": ["tests/test_parser.py::test_empty_window"],
  "pass_to_pass": ["tests/test_parser.py::test_basic", "…"],  // regression set
  "runtime": { "image": "python:3.11-slim", "timeout_s": 900, "network": false }
}
```

> **Why `base_commit` is mandatory:** a trajectory that says "in `org/repo`, fix X" is
> unreproducible in six months because the repo has moved. Pinning the commit is the
> difference between a dataset and a story.
>
> **Why `fail_to_pass` matters:** it is the definition of "fixed". It is also the thing that
> makes contamination checking possible (if a benchmark instance's `fail_to_pass` appears in
> our training data, we must know).

### 2.4 `trajectory.turns`

Each turn is one of `user | assistant | tool_result | system`.

```jsonc
"trajectory": {
  "turns": [
    { "index": 0, "role": "system", "content": "…the system prompt actually used…" },
    { "index": 1, "role": "user", "content": "Fix the off-by-one…" },
    {
      "index": 2,
      "role": "assistant",
      "content": "I'll look at the parser first.",
      "tool_calls": [
        {
          "id": "call_0",
          "name": "read_file",
          "arguments": { "path": "src/parser.py", "start_line": 40, "end_line": 90 },
          "arguments_raw": "{\"path\":\"src/parser.py\",…}"   // exact bytes the model produced
        }
      ]
    },
    {
      "index": 3,
      "role": "tool_result",
      "tool_call_id": "call_0",
      "name": "read_file",
      "content": "…file contents…",
      "truncated": false,
      "content_bytes": 4821
    }
  ],
  "patches": [
    { "path": "src/parser.py", "diff": "--- a/src/parser.py\n+++ b/src/parser.py\n@@ …" }
  ],
  "stop_reason": "completed",     // completed | max_turns | cancelled | error
  "n_turns": 14,
  "duration_s": 512
}
```

**Four rules that are easy to get wrong:**

| Rule | Why |
|---|---|
| **Store `arguments_raw` alongside parsed `arguments`** | The exact bytes the model emitted are what you re-tokenise for training. Re-serialising a parsed dict can reorder keys and change the token sequence — which silently trains the model on a distribution it will never see at inference. |
| **Store patches as diffs, not full files** | Token efficiency, and it is the form the model must learn to produce. |
| **Record `truncated: true` on any shortened tool output** | A truncated result teaches the model to *invent* the rest of the file. This is one of the most common silent poisons in agentic datasets. |
| **Keep failed turns** | See §4. |

### 2.5 `verification` — the block that makes the dataset worth anything

```jsonc
"verification": {
  "status": "pass",                  // pass | fail | error | skipped
  "verifier": "docker-runner@1.2.0",
  "verified_at": "2026-10-19T15:22:40Z",
  "tests": {
    "fail_to_pass": { "before": "FAIL", "after": "PASS" },
    "pass_to_pass": { "before": "PASS", "after": "PASS", "total": 87, "regressed": 0 }
  },
  "regressions": [],
  "sandbox": { "image_digest": "sha256:…", "network": false, "timeout_s": 900 }
}
```

> **This block is filled by the sandbox, never by the model.** If `verification.status` was
> not produced by running the tests, it must be `"skipped"` and the example must be marked
> `quality.verified = false`. A dataset where a model can write its own verification is a
> dataset that teaches confident lying. 🎯

### 2.6 `quality` — computed, never self-declared

```jsonc
"quality": {
  "verified": true,
  "score": 0.93,                     // composite 0..1
  "checks": {
    "schema_valid": true,
    "no_truncated_tool_output": true,
    "has_failure_path": true,         // ← did the agent recover from something?
    "no_secrets": true,
    "no_hallucinated_paths": true,    // every path touched must exist in the repo
    "patch_applies": true,
    "regressions": 0,
    "token_length_ok": true,
    "dedup_cluster": "cl_9f3a…"
  },
  "rejected_by": [],                 // if it survived, empty
  "filter_notes": []
}
```

### 2.7 `split`

```jsonc
"split": {
  "partition": "train",             // train | val | test
  "reason": "commit_date_before_2026-07-01",
  "split_group": "org/repo@9f2c1ab"  // ALL instances of a repo go to ONE split
}
```

> 🎯 **Two rules that prevent leakage:**
> 1. **Split by commit date**, not randomly. Random splits leak the future.
> 2. **Split by repo, not by instance.** Two instances from the same repo share utilities,
>    test helpers and idioms. One in train and one in test is leakage. `split_group`
>    enforces that an entire repo lands in one partition.
>
> Benchmark repos (SWE-bench, Terminal-Bench, LiveCodeBench, HumanEval) are **excluded from
> training entirely** and listed in `configs/dataset_config.json` → `excluded_repos`.

---

## 3. Source registry & licence manifest

Separate file: [`schemas/source-registry-v1.schema.json`](../schemas/source-registry-v1.schema.json)

```jsonc
{
  "source_id": "swe_bench_verified",
  "url": "https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified",
  "sha256": "…",
  "license": "MIT",
  "trainable": true,
  "redistributable": true,
  "role": "eval_only",          // ← never training
  "notes": "500 human-validated instances. Held out entirely."
}
```

**`role` field** — one of `train | eval_only | mixed`. `eval_only` sources are used for
evaluation and **excluded from the training pipeline by construction**, not by a filter that
someone might forget to run. 🎯

### 3.1 Licence triage table

| Licence | Trainable? | Redistributable? | Use |
|---|---|---|---|
| Apache-2.0, MIT, BSD-2/3 | ✅ | ✅ | **Preferred** |
| OpenMDW-1.1 | ✅ | ✅ | Usable, but carries an AUP — legal review first |
| CC-BY-4.0 | ✅ | ✅ (with attribution) | Usable; attribution must be carried into the dataset card |
| CC-BY-NC-4.0 | ❌ | ⚠️ | **Research only — cannot ship a commercial model trained on it** |
| Research-only (most agentic trajectory datasets) | ❌ | ⚠️ | **Do not ingest.** This is where most "20k trajectories" plans die. |
| Unknown / unspecified | ❌ | ❌ | **Reject.** Default-deny. |

> 🎯 **This table is the reason the licence gate is blocking and not advisory.** The most
> common failure in agentic dataset projects is discovering at month three that the 15k
> trajectories everyone recommended are research-only licensed. The gate moves that
> discovery to hour one.

---

## 4. The rule that separates a good dataset from a big one

### 4.1 Keep the failures

```
❌ trajectory where the agent tried 5 things and succeeded on the 5th  → often filtered as "noisy"
✅ the same trajectory, kept, with the 4 failures annotated  → THIS is the training signal
```

A model trained only on successful trajectories learns to *emit* successful-looking
sequences. It never learns to *recover*, because it has never seen a recovery.

🎯 **The dataset should target a minimum of 30% of examples containing at least one
failed-then-recovered step.** Below that, the model will not recover from tool errors, and
tool errors are what actually happens in a real agent loop.

### 4.2 Keep the honest failures too

Trajectories where the agent **failed the task and said so** are also valuable — they teach
the model to report failure honestly instead of hallucinating success. Tag them
`task.outcome = "failed_but_honest"` and keep them. 🎯 This directly trains the MenteE SWE
principle that the whole product rests on.

**Net effect:** "success rate" of a dataset is *not* a quality metric. A dataset of 100%
successful trajectories is a dataset that teaches a model to be confidently wrong.

---

## 5. Processing workflow

```
ingest ──▶ normalise ──▶ dedup ──▶ filter ──▶ verify ──▶ split ──▶ version ──▶ report
   │           │           │          │          │          │         │          │
registry   schema v1   exact+     malformed   sandbox   date+    manifest   counts by
+licence   (one fn     near-dup   unsafe      execution  repo     + sha256   reason +
gate       per source)  clusters   truncated   → f2p/p2p  group              quality
```

| Stage | Input → Output | Key parameters | Records |
|---|---|---|---|
| **ingest** | source → raw JSONL | registry lookup, licence gate | rejects: `licence_blocked` |
| **normalise** | raw → schema v1 | per-source adapter | rejects: `schema_invalid` |
| **dedup** | schema v1 → schema v1 | exact sha256 + minhash threshold 0.9 | rejects: `duplicate_exact`, `duplicate_near` |
| **filter** | schema v1 → schema v1 | ruleset in config | rejects: `truncated_tool_output`, `secret_detected`, `hallucinated_path`, `too_short`, `too_long`, `no_failure_path` (soft, not a reject) |
| **verify** | schema v1 → schema v1 | docker image, timeout, resource caps | rejects: `verification_failed`, `regression_introduced`, `sandbox_error` |
| **split** | schema v1 → schema v1 | date bounds, repo groups | marks: `partition` |
| **version** | → `dataset-v1` | manifest, sha256 per file | — |
| **report** | → `reports/dataset-v1.quality.md` | counts by reason, distribution, cost | — |

### 5.1 Quality report contents (a release artefact, not a log)

1. Input count → output count, with a **funnel table**: every stage's in/out/rejected.
2. Rejection counts **by reason**, sorted.
3. Distribution: language, task type, difficulty, n_turns, token length, source.
4. Verified vs unverified split; mean/median `n_turns` for each.
5. Percentage of examples containing a failure→recovery path.
6. Deduplication: exact and near-duplicate counts, and the largest near-dup cluster size.
7. Split sizes and the date bounds used.
8. **Cost**: wall-clock and GPU/CPU time per stage.
9. **Contamination report**: overlap against the eval source registry.

---

## 6. Storage & versioning

```
data/
├── raw/<source_id>/<sha256>/*.jsonl      # immutable, content-addressed
├── processed/<dataset_version>/
│   ├── train.jsonl   val.jsonl   test.jsonl
│   ├── manifest.json                 # dataset_version, git sha, counts, file sha256s
│   ├── rejected.jsonl                # every rejected example + reason
│   └── dedup_clusters.json
└── reports/<dataset_version>.quality.md
```

**`dataset_version` format:** `v1.0.0` — a semantic version, bumped when the schema changes,
`v1.0.1` for a content change with the same schema, `v2.0.0` for a schema change.

**Rules:**
- **Processed datasets are immutable.** Never edit a published version; publish a new one.
- **Every checkpoint records the `dataset_version`** it was trained on. Without it, results
  are not reproducible.
- **Rejected examples are kept.** They are how you debug the filters, and how you prove the
  licence gate worked.
- **JSONL is the source of truth; Parquet is an export** for analysis (`05`/pandas). Never
  train from Parquet.

---

## 7. Schema versioning

| Field | Meaning |
|---|---|
| `schema_version` | on every record, `"1.0.0"` |
| **Patch** (`1.0.1`) | additive optional field; old readers still work |
| **Minor** (`1.1.0`) | additive required field with a default; pipeline backfills |
| **Major** (`2.0.0`) | breaking change; old datasets are not silently reusable |

The pipeline **refuses to mix schema versions** in one dataset version. 🎯

---

## 8. Validation

```bash
# validate a single file
python -m src.schema.validate --file data/raw/mentee_synth_v1/part-0.jsonl

# validate + profile a whole dataset
python -m src.schema.validate --dataset data/processed/v1.0.0 --profile

# exit non-zero on any schema violation → CI gate
```

The validator is a **release gate**, not a lint. A dataset version that does not validate is
not publishable, and a training run that consumes a non-validating dataset must fail loudly
rather than train on garbage. 🎯
