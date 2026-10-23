"""Run the agent loop on ONE task end to end. The 'does the whole thing work' test.

    python scripts/run_one_task.py --task data/tasks/seed_bucket.json [--model ID]

Produces: a validated trajectory appended to datasets/collectors/<name>/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.collect.agent import AgentConfig, AgentRunner, ToolExecutor  # noqa: E402
from src.collect.appender import JsonlAppender  # noqa: E402
from src.collect.cli import _record_id, _slug  # noqa: E402
from src.collect.integrity import verify  # noqa: E402
from src.collect.normalizer import build_record  # noqa: E402
from src.collect.state import Sidecar  # noqa: E402
from src.collect.validate import TierAValidator, load_schema, make_validator  # noqa: E402


def load_key(env_name: str) -> str:
    env = os.environ.get(env_name)
    if env:
        return env
    for candidate in (Path(".env"), Path("datasets/collectors/syab/.env")):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{env_name}="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit(f"no {env_name} in env or .env")


import os  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="data/tasks/seed_bucket.json")
    ap.add_argument("--model", default="cohere/north-mini-code:free")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--collector", default="syab")
    ap.add_argument("--max-turns", type=int, default=15)
    args = ap.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    schema = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
    tier_a = TierAValidator(validator=make_validator(schema), known_tools=None)
    collector_id = _slug(args.collector)

    runner = AgentRunner(
        AgentConfig(
            base_url=args.base_url,
            api_key=load_key(args.api_key_env),
            model=args.model,
            max_turns=args.max_turns,
        ),
        ToolExecutor(task["workdir"], test_command=task.get("test_command", "")),
    )

    print(f"task   : {task['task_id']}")
    print(f"model  : {args.model}")
    print(f"workdir: {task['workdir']}")
    print("running the agent loop...\n")

    raw, meta = runner.run(task)

    print(f"\n  stop_reason : {meta['stop_reason']}")
    print(f"  turns       : {len(raw.messages)}")
    print(f"  tokens      : in={meta['total_tokens']['input']} out={meta['total_tokens']['output']}")

    rid = _record_id(Sidecar(REPO / f"datasets/collectors/{args.collector}" / "train.jsonl").load(), args.collector)
    record, problems = build_record(raw, rid)
    if record is None:
        print(f"  !! could not build a record: {problems}")
        return 1
    record["provenance"]["collected_by"] = args.collector
    record["provenance"]["generator"]["tool_name_mapping"] = {}

    # always save the raw run: a rejection without its trajectory cannot be debugged
    dbg = REPO / f"datasets/collectors/{args.collector}" / f"debug-{rid}.json"
    dbg.write_text(
        json.dumps(
            {
                "id": rid,
                "meta": meta,
                "task": {k: v for k, v in task.items() if k != "gold_patch"},
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "tool_calls": m.tool_calls,
                        "tool_results": m.tool_results,
                    }
                    for m in raw.messages
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  raw run saved to {dbg.name}")

    verdict = tier_a.validate(record)
    print(f"\n  validation  : {'PASS' if verdict.ok else 'REJECT'}")
    for reason in verdict.reasons:
        print(f"    - {reason}")

    out = REPO / f"datasets/collectors/{args.collector}" / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonlAppender(out, fsync_every=1) as app:
        if verdict.ok:
            app.append(record)
        else:
            (out.parent / "rejected.jsonl").open("a", encoding="utf-8").write(
                json.dumps({"id": rid, "reasons": verdict.reasons, "record": record}, ensure_ascii=False)
                + "\n"
            )

    print(f"\n  wrote {'ACCEPTED row' if verdict.ok else 'REJECTION record'} to {out.parent}")
    report = verify(out, deep=True, schema=schema, validator=make_validator(schema))
    print(f"  file check  : {report.summary()}")
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
