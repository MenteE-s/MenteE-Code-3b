"""Convert logged user conversations into mente-trajectory-v1 training data.

    python scripts/convert_conversations.py --in data/collected/conversations.jsonl --out data/collected/prepared.jsonl

Reads raw conversation logs from the proxy, converts them to the mente format,
validates with Tier A, and appends accepted rows to a collector's train.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def convert_conversation(conv: dict) -> dict | None:
    """Convert a logged conversation to raw trajectory dict."""
    messages = conv.get("messages", [])
    response = conv.get("response", {})
    if not messages:
        return None

    # Build common messages from the request
    common_msgs = []
    for m in messages:
        role = m.get("role", "assistant")
        content = m.get("content", "") or ""
        msg = {"role": role, "content": content}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        common_msgs.append(msg)

    # Add the assistant response
    choices = response.get("choices", [])
    if choices:
        assistant_msg = choices[0].get("message", {})
        content = assistant_msg.get("content", "") or ""
        tool_calls = assistant_msg.get("tool_calls", [])
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        common_msgs.append(msg)

    if len(common_msgs) < 2:
        return None

    return {
        "task": {
            "instruction": messages[0].get("content", "") if messages else "",
            "language": "unknown",
            "task_type": "bug_fix",
            "difficulty": "unlabelled",
            "constraints": [],
            "acceptance_criteria": [],
            "outcome": "unlabelled",
            "task_id": conv.get("session_id", ""),
        },
        "messages": common_msgs,
        "model": conv.get("model", "unknown"),
        "started_at": conv.get("started_at", ""),
        "verification": {"status": "skipped", "method": "user_feedback"},
        "generator": {
            "source_id": "user_conversation",
            "license": "user_contributed",
            "license_verified": True,
            "redistributable": False,
            "trainable": True,
            "collected_by": "community",
            "prompt_template": "user_chat",
            "teacher": {
                "model": conv.get("model", "unknown"),
                "revision": "live",
                "license": "user_contributed",
                "sampling": {},
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/collected/conversations.jsonl")
    ap.add_argument("--out", default="data/collected/prepared.jsonl")
    ap.add_argument("--append-to", default=None, help="collector train.jsonl to append accepted rows to")
    ap.add_argument("--collector", default="community")
    args = ap.parse_args()

    inp = REPO / args.inp
    if not inp.exists():
        print(f"  no conversations at {inp}")
        raise SystemExit(1)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load schema + validator
    from src.collect.validate import load_schema, make_validator, TierAValidator
    from src.collect.normalizer import GenericAdapter, build_record, RawTrajectory
    from src.collect.appender import JsonlAppender
    from src.collect.state import Sidecar
    from src.collect.cli import _record_id, _slug

    schema = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
    validator = make_validator(schema)
    tier_a = TierAValidator(validator=validator)

    collector_id = _slug(args.collector)
    sidecar = None
    appender = None
    if args.append_to:
        out_path = REPO / args.append_to
        sidecar = Sidecar(out_path).load()
        sidecar.reconcile_with_file(out_path)
        appender = JsonlAppender(out_path, fsync_every=1)
        appender.open(resume=True)
        sidecar.open()

    total = 0
    accepted = 0
    with out.open("w", encoding="utf-8") as f_out:
        for line in inp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_dict = convert_conversation(conv)
            if raw_dict is None:
                continue
            common = GenericAdapter.to_common(raw_dict["messages"])
            if not common:
                continue
            raw = RawTrajectory(
                task=raw_dict["task"],
                messages=common,
                tool_schemas=[],
                harness="user_conversation",
                model_id=raw_dict.get("model", ""),
                started_at=raw_dict.get("started_at", ""),
                verification=raw_dict.get("verification", {}),
                generator=raw_dict.get("generator", {}),
            )
            rid = _record_id(sidecar, collector_id) if sidecar else f"conv_{total:06d}"
            record, problems = build_record(raw, rid)
            if record is None:
                continue
            total += 1
            if collector_id:
                record["provenance"]["collected_by"] = collector_id

            verdict = tier_a.validate(record)
            if verdict.ok:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
                if appender and sidecar:
                    appender.append(record)
                    sidecar.add_id(record["id"])
                    sidecar.add_task((record.get("task") or {}).get("task_id", ""))
            else:
                pass  # rejected, skip

    if appender:
        appender.close()
    if sidecar:
        sidecar.close()

    print(f"  converted: {total}, accepted: {accepted} -> {out}")
    if args.append_to:
        print(f"  appended accepted rows to {args.append_to}")


if __name__ == "__main__":
    raise SystemExit(main())