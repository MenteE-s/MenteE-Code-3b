# Serving, API Layer & the 10-Day Public Beta

> Everything here is sized for a **bootstrapped project**: one 24 GB GPU, one process,
> a gateway that is ~200 lines, and a kill switch that is one env var.

---

## 1. Serving stack

```bash
# Student: MenteE Code 3B, AWQ Q4, 32K context, on one 24 GB card
vllm serve MenteEAI/mentee-code-3b-v0.1 \
  --host 0.0.0.0 --port 8000 \
  --quantization awq \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --served-model-name mentee-code-3b
```

| Flag | Why |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | ✅ The documented way to serve Qwen3-Coder tool calls. Without these, tool calls come back as raw text and the harness cannot parse them. **This is the single most important flag pair in the whole file.** |
| `--max-model-len 32768` | The Q4 weights are 18.6 GB ✅; the rest of 24 GB is KV cache. 32K is the honest ceiling at this VRAM. |
| `--max-num-seqs 16` | Concurrency target. Tune from measured throughput. 📐 |
| `--gpu-memory-utilization 0.90` | Leave headroom; OOM mid-beta is worse than slightly lower concurrency. |
| `--served-model-name mentee-code-3b` | The id the gateway exposes. **This is the id OpenCode will request.** |

### 1.1 Why vLLM

| Option | Verdict |
|---|---|
| **vLLM** ✅ | OpenAI-compatible out of the box, `--tool-call-parser qwen3_coder` ✅, continuous batching, best-supported option for this model family. **Chosen.** |
| SGLang | Valid alternative ✅ (requires ≥ 0.5.8 for Coder-Next ✅). Keep as a fallback if vLLM has a Gated DeltaNet issue. |
| llama.cpp / Ollama | Great for local dev, weaker on concurrent serving. Use locally, not for the beta. |
| Managed endpoint (Together, Fireworks, etc.) | 📐 Good fallback if our own GPU ops become a burden — but it moves the margin to someone else and changes the cost model. Revisit at v0.2. |

---

## 2. Agent integration

### 2.1 OpenCode (primary target)

**OpenCode v1 config** ✅ — `provider` map, `options.baseURL`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mentee": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MenteE",
      "options": {
        "baseURL": "https://api.menteeai.org/v1",
        "apiKey": "{env:MENTEE_API_KEY}"
      },
      "models": {
        "mentee-code-3b": { "name": "MenteE Code 3B" }
      }
    }
  },
  "model": "mentee/mentee-code-3b"
}
```

**OpenCode v2 config** ✅ — note the renamed keys: `providers` (plural), `package` (not `npm`),
`settings` (not `options`), and a different adapter path:

```jsonc
{
  "$schema": "https://opencode.ai/v2/config.json",
  "providers": {
    "mentee": {
      "name": "MenteE",
      "package": "@opencode/ai/providers/openai-compatible",
      "env": ["MENTEE_API_KEY"],
      "settings": { "baseURL": "https://api.menteeai.org/v1" },
      "models": {
        "mentee-code-3b": { "modelID": "mentee-code-3b", "name": "MenteE Code 3B" }
      }
    }
  },
  "model": "mentee/mentee-code-3b"
}
```

### 2.2 Integration gotchas — all verified, all bite

| # | Gotcha | Detail ✅ | Fix |
|---|---|---|---|
| **1** | **`baseURL` must end in `/v1`** | Not `/v1/chat/completions` ✅ | `https://api.menteeai.org/v1` |
| **2** | **v1 vs v2 key names** | v1 = `provider` / `npm` / `options.baseURL`; v2 = `providers` / `package` / `settings.baseURL` ✅ | Match the installed version. Check with `opencode --version` ✅ |
| **3** | **v1 config in v2 → request to `undefined/chat/completions`** | A known bug: v2 silently ignores the v1-style `options.baseURL` and builds an unparseable URL ✅ | Use the v2 shape if running v2. **Test this on day 1.** |
| **4** | **`reasoningEffort: none` → upstream 400** | Some upstreams reject the parameter ✅ | Don't send it. Our gateway should strip it. |
| **5** | **v2's native `openai-compatible` package is unregistered at runtime** | The documented v2 native package may not actually load ✅ | Use `@ai-sdk/openai-compatible` in v1; in v2 verify the adapter loads before relying on it. |
| **6** | **Model id must match exactly** | The key in `models` is the id sent in the `model` field ✅ | The gateway's `/v1/models` must list the same string. |
| **7** | **Not every agent uses the same protocol** | The blueprint says this ✅ — and it is true. Roo Code requires native tool calling ✅; Kilo Code uses `GET /v1/models` to populate the model list ✅. | Test each agent explicitly. **Do not assume.** |

> 🎯 **Items 1–3 are the ones that will cost you a day.** Put "connect OpenCode to a local
> vLLM and complete one trivial task" as **Day 0** work, before any training. It is a
> 20-minute test that de-risks the entire project.

### 2.3 Second agent (required by the blueprint)

**Test with at least one non-OpenCode agent.** Candidates:

| Agent | Why | Note |
|---|---|---|
| **MenteE SWE** (ours) ✅ | Dogfooding; we control both ends | The natural first choice — it's in the sibling repo |
| Kilo Code | Uses `GET /v1/models` ✅ — a different discovery path | Proves the `/v1/models` endpoint works |
| Roo Code | Requires native tool calling ✅ — a different tool-call path | Proves our tool-call format is not OpenCode-specific |
| Cline | Different config surface | |

> 🎯 **Pick one that exercises a different code path than OpenCode.** Testing two agents
> that both use `@ai-sdk/openai-compatible` proves less than testing one that doesn't.

---

## 3. API gateway

A thin layer in front of vLLM. **~200 lines, not a platform.**

### 3.1 Responsibilities

| Feature | Implementation |
|---|---|
| **Auth** | Bearer token → hashed key → user record. Keys in `~/.mentee` style config, never in code. |
| **Model routing** | `/v1/models` returns exactly the ids we serve ✅ (Kilo Code depends on this) |
| **Streaming** | Pass through SSE from vLLM unchanged |
| **Tool calls** | Pass through; the parser flag on vLLM does the work ✅ |
| **Timeouts** | 120 s default; 600 s for long agent turns (configurable per key) |
| **Rate limit** | Token-bucket per key: requests/min **and** tokens/min |
| **Usage accounting** | Prompt tokens, completion tokens, cached tokens → per key, per day, persisted |
| **Quota** | Per-key daily token cap + daily request cap. **Exceeded → HTTP 429 with a plain explanation.** |
| **Spend cap** | A global daily USD cap for the whole service |
| **Kill switch** | One env var flips the gateway to refuse all traffic and return 503 ✅ |
| **Parameter scrubbing** | Strip `reasoningEffort` if it causes 400s ✅ (gotcha #4) |
| **Logging** | Structured JSON: request id, key id, model, tokens, latency, status. **No prompt bodies by default.** |

### 3.2 The kill switch (write it before you need it)

```bash
# .env
MENTEE_SERVICE_ENABLED=true
MENTEE_MAX_DAILY_SPEND_USD=25
MENTEE_MAX_CONCURRENT=16
```

When `MENTEE_SERVICE_ENABLED=false`, the gateway returns `503` immediately without touching
vLLM. Flipping it requires one `docker compose restart gateway` — under 5 seconds. 🎯

> **Test the kill switch before the beta, not during an incident.** A drill costs 10 minutes.

---

## 4. The 10-day public beta

### 4.1 Launch gates (all must be green)

| Gate | Requirement |
|---|---|
| G1 | Tool-call conformance ≥ 99.5% at serving quantisation |
| G2 | In-harness task completion ≥ base model, same prompts |
| G3 | Cost per successful task ≤ $0.10 |
| G4 | No regression > 2 pts on LiveCodeBench / HumanEval+ |
| **Licence** | Apache-2.0 attribution ✅, dataset licences cleared ✅, base model cited ✅ |
| **Security** | E5 safety suite passes ✅; sandbox isolation verified ✅ |
| **Privacy** | Consent flow, retention policy, redaction, deletion, opt-out — all live and tested ✅ |
| **Cost** | Daily spend cap + kill switch tested ✅ |

### 4.2 Free-beta allowance

| Tier | Allowance |
|---|---|
| Free | **500k tokens/day**, max 20 requests/min |
| Internal / dogfood | 2M tokens/day |
| Paid (post-beta) | TBD from beta data |

**Rationale:** a daily token cap means a single user cannot consume the daily budget, and the
whole beta's exposure is bounded by `MAX_DAILY_SPEND_USD` regardless of usage. 🎯

### 4.3 Day-by-day beta plan

| Day | Focus | Output |
|---|---|---|
| **D1** | Invite 10 users. 1:1 onboarding. Watch every session. | First-day transcripts |
| **D2** | Watch. Fix format issues. | Bug list |
| **D3** | Open to 50 users. | Usage baseline |
| **D4** | First feedback round. | Curated candidate set |
| **D5** | Midpoint review. Check cost trajectory vs cap. | Cost report |
| **D6** | Fix the top 3 reported issues. | Patch release |
| **D7** | Open to 100 users. | Usage at scale |
| **D8** | Analyse: completion rate, cost/task, failure modes. | Metrics dashboard |
| **D9** | Write the model card + honest limitations. | Model card draft |
| **D10** | Publish results. Decide v0.2. | Public report |

### 4.4 What to measure every single day

| Metric | Why it matters |
|---|---|
| Task completion rate | The product metric |
| **Cost per successful task** | The business metric — re-measure at real concurrency |
| Median / p95 latency | User-perceived quality |
| Tokens per resolved task | Provider-independent efficiency |
| Tool-call parse failure rate | If this is non-zero, the product is broken |
| Rate of "claimed success, actually failed" | **The honesty metric.** If it rises, the data is wrong. |
| Prompt injection incidents | Security |
| Unique repos touched | Breadth of real use — the best guide for the next data batch |
| Daily spend vs cap | Cost control |

> 🎯 **The "unique repos touched" number is the most valuable output of the beta.** It tells
> you exactly which repos, languages and task types to collect next. This is how the beta
> feeds v0.2's dataset. 👈

---

## 5. Privacy, consent & data governance

### 5.1 The default: nothing is training data

> 🎯 **Every interaction is opt-in for training, explicitly, per session, with an
> unambiguous affirmative action.** No dark patterns, no "by using this service you agree to
> our terms may include your data in training", no pre-ticked boxes.

| Rule | Implementation |
|---|---|
| **Explicit consent** | A `/share` command, or a UI toggle, **off by default**. Consent state is stored per user and per session. |
| **Clear disclosure** | At first use, in the first screen: what may be collected, what won't, how to opt out, how to delete. |
| **Granular scope** | Consent for "prompts" and consent for "code" and consent for "tool outputs" are **separate** choices. |
| **Easy opt-out** | One command, immediate effect, applies retroactively to future collection. |
| **Retention limit** | Raw sessions deleted after 30 days. Curated candidates retained until the v0.2 training run completes, then deleted. |
| **Deletion on request** | A user command deletes their data — raw, curated, and any derived artefacts. Implemented and **tested** before launch. |
| **No auto-retrain** | Feedback goes to a human-reviewed candidate set. A model is never retrained on live traffic without review. |
| **Redaction before retention** | Secrets, keys, tokens, PII scrubbed before anything is stored in the candidate set. |

### 5.2 Redaction

Before an example enters the candidate set, scrub:

- API keys, tokens, private keys, connection strings (regex + entropy heuristics)
- `.env` file contents, credential files
- Absolute home paths and usernames
- Email addresses, phone numbers (PII)
- Anything matching a high-entropy string ≥ 24 chars

**Run the redactor on the training dataset too** — QC check #10 ✅. A leaked key in the
training set gets published with the weights. 🎯

### 5.3 Disclosure

The public beta page must state, before sign-up:

1. What data is collected, and the exact retention period.
2. That training use is **opt-in and off by default**.
3. How to opt out, and how to delete.
4. That prompts and code may contain proprietary or confidential material — and therefore
   **users should not use the beta for proprietary code** unless they have explicitly opted in.
5. Who to contact.
6. The model's known limitations — honestly.

> 🎯 **Point 4 is the one that gets forgotten and it is the one that causes the most harm.**
> A developer who pastes proprietary code into a service that might retain it has been
> misled, whatever the terms say.

---

## 6. Sandbox security

The beta executes model-generated code. This is the highest-risk part of the operation.

| Control | Requirement |
|---|---|
| **Disposable container** | Fresh per run. Never reused. |
| **No network** | `network=none`. A test that needs the internet is not a valid verification. |
| **No host secrets** | Never mount `$HOME`, never pass API keys into the container. |
| **Read-only source** | The repo is mounted read-only except the workdir. |
| **Resource caps** | CPU, memory, disk, wall-clock, process count, output size. |
| **Process limits** | `pids-limit`, no privilege escalation, no host namespaces. |
| **Log sanitisation** | Container output is scrubbed before logging. |
| **Timeout + kill** | Hard kill on wall-clock; the container is destroyed, not paused. |
| **Escaped-content handling** | Everything the model produces is treated as untrusted. |

> 🎯 **The model can attempt anything.** The runtime blocks it, and the model recovers. A
> blocked command that the model retries 20 times is a cost problem; a command that *succeeds*
> is a security problem. Test the second case in E5.

---

## 7. Rollback

| Level | Trigger | Action | Time |
|---|---|---|---|
| **L1** | Model regression after a patch | Serve the previous checkpoint (keep 3) | < 5 min |
| **L2** | Cost anomaly | Enable the spend cap / reduce concurrency | < 1 min |
| **L3** | Security concern | Kill switch → 503 | < 1 min |
| **L4** | Licence issue | Unpublish from HF, retract the release | hours |
| **L5** | Full stop | Disable the service, keep the data, publish an incident note | < 1 min |

**Checkpoint retention:** keep the last 3 adapters + the last 2 merged-and-quantised models.
Rolling back is a config change, not a rebuild. 🎯

**The model card and the dataset card are published atomically with the weights.** If we
retract the weights, we retract the cards. A model card describing a model that no longer
exists is worse than no model card.

---

## 8. Observability

| Signal | Where | Alert when |
|---|---|---|
| Request rate, latency (p50/p95/p99) | gateway | p95 > 30 s |
| Error rate by status | gateway | > 2% over 5 min |
| **Tool-call parse failures** | gateway + vLLM | **> 0** — any non-zero is a product defect |
| Tokens in / out, cached | gateway | — (accounting) |
| Daily spend | gateway | > 80% of cap |
| GPU utilisation / VRAM | vLLM | < 40% (wasted money) or OOM events |
| Task completion rate | eval harness | weekly |
| Kill-switch availability | monitoring | never |

**Everything the beta produces is a JSON artefact**, aggregated into a daily report. The
v0.2 decision at D10 is made from data, not from impressions. 🎯
