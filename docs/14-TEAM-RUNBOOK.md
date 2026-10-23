# Team Runbook — Running the Collector

> **Your job:** run the collector on your machine against your own API key, collect
> trajectories into **your own folder**, and push. You do **not** merge — Syab does that
> centrally.
>
> Total time: ~15 minutes of setup, then the collector runs unattended.

---

## 1. One-time setup (~10 minutes)

```bash
git clone <repo> && cd mentee-code-3b

# Python 3.12+ is required (the repos we collect from use modern syntax).
# On Windows, create a venv once and use it everywhere below:
py -3.12 -m venv .venv312
.venv312\Scripts\python.exe -m pip install -r requirements-collect.txt

# 1. claim your collector name (use your own name, lowercase)
.venv312\Scripts\python.exe -m src.collect init --collector <yourname>
```

This creates **only your** files:

```
configs/collectors/<yourname>.yaml      your config
datasets/collectors/<yourname>/         your output folder  <- you push this
datasets/collectors/<yourname>/.env     your keys           <- never pushed
```

Nobody else writes to your folder, so there are no naming conflicts and no git
merge conflicts.

2. Put your key in `datasets/collectors/<yourname>/.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

It is gitignored — **never** put a key anywhere else, and **never** paste it into
the chat, a ticket or a screenshot. The CLI loads this file automatically on
every command, so no `export` is needed.

3. Get your task shard from Syab (see §4) into `data/tasks/`, then build your
local worktree for each task (**run this once per shard — it makes no API
calls**, it only checks out code and verifies the tests fail):

```bash
.venv312\Scripts\python.exe scripts/prepare_tasks.py --in data/tasks/shard-2-of-4.jsonl --out data/tasks/prepared-<yourname>.jsonl
```

Point your config's `source.tasks_path` at your `prepared-<yourname>.jsonl`.

---

## 2. Run it

```bash
# dry run first - generates ONE trajectory and prints it, costs almost nothing
.venv312\Scripts\python.exe -m src.collect run --config configs/collectors/<yourname>.yaml --dry-run

# real run, capped for the first day
.venv312\Scripts\python.exe -m src.collect run --config configs/collectors/<yourname>.yaml --limit 20
```

Then check it worked:

```bash
.venv312\Scripts\python.exe -m src.collect verify --path datasets/collectors/<yourname>/train.jsonl
.venv312\Scripts\python.exe -m src.collect tail   --path datasets/collectors/<yourname>/train.jsonl -n 3
.venv312\Scripts\python.exe -m src.collect report --path datasets/collectors/<yourname>/train.jsonl
```

The agent source also writes `debug-<id>.json` for every run it rejects, so a
failed row can always be inspected or recovered with
`scripts/recover_from_debug.py`.

The report prints the three numbers that decide whether your run is any good:

| Metric | Target |
|---|---|
| `acceptance_rate` | how many rows survived the 13 checks |
| `verified_rate` | **≥ 80%** |
| `failure_path_rate` | **≥ 30%** |

If `verified_rate` is low, your sandbox or harness is misconfigured — **stop and fix
it before collecting more.** Bad data is worse than no data.

---

## 3. Push

```bash
git add datasets/collectors/<yourname>/
git commit -m "collect: <yourname> - N trajectories"
git push
```

Only your folder. Only your data. Syab merges.

**Do not** commit: `.env`, `data/repos/`, `data/workdirs/`, or anything under
`data/raw/`. All gitignored already.

---

## 4. Task shards — why you got different tasks

All four of you working the same tasks would waste 75% of the quota and produce
near-identical trajectories, which the merge would just throw away.

So tasks are pre-split into **disjoint shards** — every task belongs to exactly one
person:

```bash
# Syab runs this once, then hands out the files:
python -m src.collect partition --tasks data/tasks/prepared.jsonl --collectors 4

#   shard-1-of-4.jsonl
#   shard-2-of-4.jsonl
#   shard-3-of-4.jsonl
#   shard-4-of-4.jsonl
```

Point your config's `source.tasks_path` at your own shard. The split is
deterministic — re-running it always produces the same assignment.

---

## 5. Rate limits (OpenRouter free tier)

| | Requests/min | Requests/day |
|---|---|---|
| Never paid | 20 | **50** |
| Ever bought ≥ $10 credits | 20 | **1,000** |

One trajectory ≈ 15–40 requests, so:

- **50/day** → 1–3 trajectories/day. Enough for testing, nothing more.
- **1,000/day** → 25–60 trajectories/day.

**A trajectory needs 15–40 requests in a row.** If you hit a 429 mid-run, the run
is lost. Expect this on free models — that is why the collector validates before
appending, and why a lost run costs you rate limit, not dataset quality.

**Failed requests still count** against your daily allowance. Cap your retries.

Your current tier is visible at any time:

```bash
python -c "import os,json,urllib.request as u; k=os.environ['OPENROUTER_API_KEY']; print(json.loads(u.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+k}).read())['data']['free_model_daily_requests'])"
```

---

## 6. When something looks wrong

| Symptom | Meaning | Action |
|---|---|---|
| `verified_rate` < 50% | your sandbox or workdir is broken | stop; fix setup; delete your file and re-run |
| `failure_path_rate` ≈ 0% | you are filtering out failures | do not filter during collection — failures are the valuable rows |
| `integrity FAILED` in the log | the file on disk diverged from what the collector wrote | **stop immediately**, keep the file, tell Syab. Do not re-run over it. |
| HTTP 429 everywhere | free-tier rate limit | wait, or move to a smaller `--limit` |
| every row rejected `no_truncated_output` | the harness truncates tool output | that check is correct; the harness config needs a bigger output budget |
| all rows `duplicate_content` at merge | you collected the same tasks as someone else | you used the wrong shard |

---

## 7. The one rule

> **Never report success you have not verified, and never commit a key.**

Everything else is recoverable.

---

*Setup questions: `docs/13-COLLECTOR-TOOLKIT.md`. What the checks do: §5. Why the
tool schema is varied per trajectory: `docs/01-THESIS-300B-CLASS.md` §2.5.*
