# MenteE Code 3B

**A 3B-activated-parameter coding model for autonomous coding agents.**

> MenteE Code 3B is a code-specialised language model that lives in an agent's terminal. It
> investigates repositories, edits files, runs tests, recovers from failures, and reports
> honestly whether the task succeeded. It is built for OpenCode and similar harnesses, and
> served over a standard OpenAI-compatible API.

**Status: planning / pre-data.** No training has been run. No benchmark results have been
produced. Every external fact in the documentation was verified against a primary source on
**2026-10-19**.

---

## What it is

| | |
|---|---|
| **Base** | `Qwen/Qwen3-Coder-30B-A3B-Instruct` â€” 30.5B total / **3.3B active**, Apache-2.0 |
| **Teacher** | `Qwen/Qwen3-Coder-Next` â€” 80B total / **3B active**, Apache-2.0, self-hosted |
| **Method** | QLoRA supervised fine-tuning â†’ rejection fine-tuning (RFT) on execution-verified trajectories |
| **Serving** | vLLM, OpenAI-compatible `/v1/chat/completions`, AWQ Q4 on **one 24 GB GPU** |
| **Moat** | Our own execution-verified agent trajectory dataset â€” the only thing here that is not a commodity |
| **Cost** | ~$30 to prove the idea Â· ~$1,020 to ship v0.1 + a 10-day beta Â· **~$0.02 per successful task** |

---

## The thesis in one paragraph

A 3B-activated Mixture-of-Experts model **can** behave like a 300B model on agentic coding â€”
this is demonstrated, not hypothetical: `Qwen3-Coder-Next` activates 3B parameters per token
and reports **70.6 on SWE-bench Verified** and **36.2 on Terminal-Bench 2.0** âœ…. Five
mechanisms make it happen: a sparse knowledge store (total â‰« active), cheap long-context
attention, execution-verified post-training, long-horizon recovery training, and a native
tool-call format. **We inherit the first, second and fifth from the base model. The third and
fourth can only come from our data.** That is the entire project, and it is why the data
pipeline matters more than the training script.

Full argument: [`docs/01-THESIS-300B-CLASS.md`](docs/01-THESIS-300B-CLASS.md)

---

## Documentation

Start at **[`docs/README.md`](docs/README.md)** for the index and the 60-second version.

| | |
|---|---|
| [`00` Executive Summary](docs/00-EXECUTIVE-SUMMARY.md) | The direction, the five release gates, the kill switch |
| [`01` The 300B-Class Thesis](docs/01-THESIS-300B-CLASS.md) | How a 3B model behaves like a 300B model |
| [`02` Base Model Selection](docs/02-BASE-MODEL-SELECTION.md) | Candidate matrix, licences, training traps |
| [`03` Architecture](docs/03-ARCHITECTURE.md) | Pipeline, components, the model-vs-runtime boundary |
| [`04` Data Schema](docs/04-DATA-SCHEMA.md) | Canonical format, licence manifest, dedup, splits |
| [`05` Data Collection Playbook](docs/05-DATA-COLLECTION-PLAYBOOK.md) | ðŸŽ¯ **The weekend job** |
| [`06` Training](docs/06-TRAINING.md) | Stage 0 â†’ pilot â†’ scale â†’ RFT, with configs |
| [`07` Evaluation](docs/07-EVALUATION.md) | The benchmark suite, G1â€“G5, contamination control |
| [`08` Serving & Beta](docs/08-SERVING-AND-BETA.md) | vLLM, OpenCode, gateway, privacy, sandboxing |
| [`09` Cost Model](docs/09-COST-MODEL.md) | Three estimates, verified prices, cost per task |
| [`10` Roadmap & Backlog](docs/10-ROADMAP-AND-BACKLOG.md) | Day-by-day, critical path, P0/P1/P2 |
| [`11` Risks & Fallbacks](docs/11-RISKS-AND-OPEN-QUESTIONS.md) | Every risk with trigger, mitigation, fallback |
| [`12` Open Questions](docs/12-OPEN-QUESTIONS.md) | What was **not** verified |
| [`13` The Collector Toolkit](docs/13-COLLECTOR-TOOLKIT.md) | The code: generate â†’ validate â†’ append, the append guarantees, the tool-schema shuffler |
| [\ Team Runbook](docs/14-TEAM-RUNBOOK.md) | Per-collector setup, folders, task shards, pushing, rate limits |

---

## The collector

```bash
pip install -r requirements-collect.txt

# prove the plumbing on a replay file (no GPU, no agent needed)
python -m src.collect run --config configs/collect_config.replay.yaml

# check the file is what we think it is
python -m src.collect verify --path data/processed/v1.0.0/train.jsonl
```

The loop is **generate one trajectory â†’ validate it â†’ append it â†’ repeat**. The dataset
file is only ever appended to â€” never truncated, never rewritten, never reopened for a
"save" â€” and a widening integrity schedule checks the file against the collector's own
belief as it goes.

---

## Repository layout

```
docs/      the documentation set (start here)
schemas/   canonical JSON Schema for trajectories + the source registry
src/collect/  the collector toolkit
configs/   collector config (one per team member)
tests/     the append, check, schedule, merge and end-to-end tests
datasets/  per-collector output (tracked - this is what the team pushes)
```

The training pipeline, the evaluator and the gateway are **not yet written** â€” see
[`docs/10-ROADMAP-AND-BACKLOG.md`](docs/10-ROADMAP-AND-BACKLOG.md) for the build order.

---

## Next action

**[`docs/05-DATA-COLLECTION-PLAYBOOK.md`](docs/05-DATA-COLLECTION-PLAYBOOK.md) Â§7 â€” the weekend
checklist.** The critical path is four steps: sandbox â†’ 20 trajectories â†’ 500 verified â†’
pilot â†’ gate.

---

*MenteE AI Â· menteeai.org Â· Apache-2.0 lineage. Model weights to be released under Apache-2.0
with the base model's attribution retained, as required.*
