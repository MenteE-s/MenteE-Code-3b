# The Collector Toolkit — `src/collect`

> One loop: **generate one trajectory → validate it → append it → repeat.**
>
> The dataset file is only ever appended to. No batching, no rewriting, no reopening
> for a "save".

---

## 1. Quick start

```bash
pip install -r requirements-collect.txt

# 1. Prove the plumbing on a replay file (no GPU, no agent needed)
python -m src.collect run --config configs/collect_config.replay.yaml

# 2. Check the file is what we think it is
python -m src.collect verify --path data/processed/v1.0.0/train.jsonl

# 3. See the collector's own state
python -m src.collect status --path data/processed/v1.0.0/train.jsonl

# 4. Look at the last few rows
python -m src.collect tail --path data/processed/v1.0.0/train.jsonl -n 5

# 5. Read the run report
python -m src.collect report --path data/processed/v1.0.0/train.jsonl
```

For the live path, point `source.kind: harness` at a running agent and set
`OPENCODE_SERVER_PASSWORD` (or whichever env var your config names).

---

## 2. Module map

| Module | Responsibility |
|---|---|
| `appender.py` | Append-only JSONL writer. Never truncates, never leaves a partial line, fsyncs, keeps counters. |
| `integrity.py` | `verify()` — compares belief vs. disk. `IntegritySchedule` — the widening check cadence. |
| `state.py` | The sidecar: believed counters, run id, seen trajectory ids. Atomic writes. |
| `checks.py` | The 13 Tier A checks. One function each, so each can be tested in isolation. |
| `validate.py` | `TierAValidator` (the gate) and `DeepVerifier` (Tier B: sandbox + patch-apply). |
| `normalizer.py` | Harness adapters → canonical schema v1. |
| `schema_tools.py` | The tool-schema shuffler (§4). |
| `source.py` | `HarnessSource` (live agent) and `ReplaySource` (import). |
| `proxy.py` | Logging OpenAI-compatible passthrough — captures the model's exact bytes. |
| `report.py` | Running stats, rejection histogram, verified/failure-path rates. |
| `budget.py` | Row / minute / token ceilings. |
| `config.py` | Config loading, merged over `DEFAULTS`. |
| `cli.py` | `run` / `verify` / `status` / `tail` / `report`. |

---

## 3. The append guarantees

These are the properties that make an unattended multi-hour run safe. Each one has a
test in `tests/test_appender.py`.

| Guarantee | Mechanism | Test |
|---|---|---|
| **Never truncates** | the file is opened in binary **append** mode only; a second run without `resume=True` is refused | `test_refuses_to_open_existing_without_resume` |
| **Never a half-line** | one `write()` of a complete payload per record; a trailing partial line is healed on open | `test_heals_partial_trailing_line` |
| **Survives a hard kill** | the same, exercised through a real `os._exit(9)` mid-write | `test_survives_a_hard_kill` |
| **Never loses acknowledged rows** | `flush()` every record, `fsync()` every N and at every checkpoint | `test_append_creates_and_counts` |
| **Never silently writes nothing** | the sidecar holds the writer's belief; every checkpoint compares it to disk and **aborts** on mismatch | `test_belief_mismatch_is_detected` |
| **Never duplicates** | trajectory ids are tracked in the sidecar's `.ids` file | `test_duplicate_ids_are_detected_by_the_sidecar` |
| **Never interleaves writers** | an advisory lock on `<path>.lock` | — |

### 3.1 The widening integrity schedule

The check cadence starts tight and widens, with jitter so the check never lands on a
predictable beat:

```
row 10 → ~40 → ~85 → ~145 → ~235 → every ~200 (±30% jitter)
```

Every 5th check re-parses **every** line; the rest check the tail only. The final check
at the end of a run is always a full re-parse.

```yaml
integrity:
  first_check_at: 10
  gaps: [30, 45, 60, 90]
  steady_every: 200
  jitter_ratio: 0.30
  full_rescan_every: 5
  on_failure: abort
```

**On failure the run aborts.** Continuing to append to a file that has diverged from what
we believe we wrote is the exact failure this whole mechanism exists to prevent.

---

## 4. The tool-schema shuffler — how we avoid a single-harness model

> The **wire format** is fixed (it comes from the base model). The **tool schema** is a
> request parameter. Full explanation: `01-THESIS-300B-CLASS.md` §2.5 (M5.1).

If every training trajectory uses the same tool names, the model learns a lookup table
instead of learning to read a schema. The shuffler varies the schema for free, because
the schema lives in the request:

- **renames tools** — `read_file` → `read` / `view_file` / `open_file` / …
- **renames parameters** — `path` → `file_path` / `filename` / … (and keeps `required` in sync)
- **drops tools** — so the advertised set varies in size
- **adds tools** — from a pool of extras, so the set varies in shape

```yaml
source:
  tool_schema_shuffle:
    enabled: true
    probability: 0.35        # fraction of trajectories that get a variant
    rename_probability: 0.8
    rename_params_probability: 0.7
    drop_probability: 0.25
    add_probability: 0.35
```

Variants are **seeded from the task id**, so a run is reproducible: the same task always
produces the same schema variant.

Guarantees covered by `tests/test_integrity_and_tools.py`:
- deterministic for a given seed
- **never produces duplicate tool names**
- **never empties the tool set**
- **never renames a parameter without renaming it in `required`**

---

## 5. The validation tiers

### 5.1 Tier A — the per-row gate (every row, < 5 ms)

| # | Check | Rejects on |
|---|---|---|
| 1 | `schema` | any field violation against `schemas/mentee-trajectory-v1.schema.json` |
| 2 | `turn_sequence` | non-contiguous indexes, illegal role, `n_turns` mismatch |
| 3 | `tool_call_pairing` | a call with no result, a result with no call, duplicate ids |
| 4 | `arguments_raw` | `arguments_raw` missing, unparseable, or ≠ the parsed `arguments` |
| 5 | `tool_names` | a tool that was not advertised |
| 6 | `no_truncated_output` | **any** truncated tool result — teaches hallucination |
| 7 | `no_secrets` | credential-shaped strings (placeholders are allowed) |
| 8 | `no_tool_loop` | the identical call repeated ≥ 3× |
| 9 | `length_bounds` | too short to teach anything, or too long for the context |
| 10 | `provenance` | `trainable != true` or `license_verified != true` — **the licence gate** |
| 11 | `verification_consistency` | `status: pass` with no `fail_to_pass`; a regression; `skipped` while claiming verified |
| 12 | `no_hallucinated_paths` | a patched path that is not in the repo |
| 13 | `honest_conclusion` | the agent claimed success while verification says it failed |

> 🎯 **Check 4 is subtle and important.** `arguments_raw` must be the *exact bytes* the
> model emitted. Re-serialising a parsed dict can reorder keys, which shifts the token
> sequence — so the model would be trained on a distribution it never sees at inference.
> The logging proxy (`proxy.py`) exists to capture the real bytes.

> 🎯 **Check 13 is compatible with keeping failures.** A `failed_but_honest` row is one
> where the agent *did not* claim success. A row that claims success on a failed run is
> dishonest and is rejected.

### 5.2 Tier B — deep verification (sampled)

Runs the real code: applies the patch to the base commit, executes the test command,
and computes `fail_to_pass` / `pass_to_pass` evidence. On by default for **2%** of rows,
*before* they are appended — so a sampled row that fails never lands in the file.

```yaml
validation:
  tier_b_sample_rate: 0.02
  require_verified: false    # true = only append rows that passed the sandbox
```

`require_verified: true` is the strict mode for the release pass: only rows with a green
sandbox run are appended.

---

## 6. The logging proxy

`proxy.py` sits between the agent and the inference server and passes traffic through
untouched while recording it. It gives us three things the harness cannot:

1. **the exact bytes** the model emitted for each tool call → `arguments_raw` ✅
2. **the exact rendered messages** the model saw → training re-renders faithfully
3. **the exact tool schema array** that was offered → training uses the same schemas

```yaml
source:
  proxy:
    enabled: true
    listen_port: 8010
    log_dir: data/raw/proxy_logs
```

Point the agent's `baseURL` at the proxy; the proxy forwards to the teacher.

---

## 7. What a run produces

```
data/processed/v1.0.0/
├── train.jsonl            # the accepted trajectories (append-only)
├── train.jsonl.state.json # believed counters, run id, last id, rejections
├── train.jsonl.ids        # one trajectory id per line (duplicate detection)
├── train.jsonl.lock       # advisory lock
├── rejected.jsonl         # every rejected row + the reason it was rejected
└── train.report.json      # the run report
```

**Rejected rows are kept, not deleted.** They are how you debug the filters, and how you
prove the licence gate worked.

The report answers the three questions that matter:

| Metric | Target |
|---|---|
| `acceptance_rate` | how strict the filters are |
| `verified_rate` | ≥ 80% |
| `failure_path_rate` | ≥ 30% — this is M4, the signal that makes it an *agent* |
| `rejections_by_check` | which filter is eating the data |

---

## 8. Adding a harness

1. Write an adapter with a `to_common(messages, raw_args_by_call) -> list[CommonMessage]`
   classmethod (`normalizer.py` has two examples: `OpenCodeAdapter`, `GenericAdapter`).
2. Add a `RawTrajectory` producer to `source.py` (or use `ReplaySource` and export in the
   common shape).
3. Nothing else changes. The validator, the appender, the integrity checks and the
   reporting are all harness-agnostic.

---

## 9. Tests

```bash
python -m pytest tests/ -q
```

| File | Covers |
|---|---|
| `test_appender.py` | the append guarantees, including a real hard kill |
| `test_checks.py` | every Tier A check with a passing and a failing fixture; the record builder |
| `test_integrity_and_tools.py` | the schedule; the shuffler's determinism and safety |
| `test_collector_loop.py` | the whole loop end to end, including resume-after-crash |
