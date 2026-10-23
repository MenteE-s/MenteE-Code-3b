"""Preprocess external trajectory datasets into mente-trajectory-v1 format.

    python scripts/preprocess_external.py --in data/external --out data/external/prepared.jsonl

Reads raw downloaded datasets and converts them to the mente format.
Handles SWE-agent and OpenHands trajectory schemas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def convert_swe_agent(row: dict) -> dict | None:
    """Convert a SWE-agent trajectory row to mente format."""
    messages = []
    # SWE-agent stores trajectory as a list of steps
    traj = row.get("trajectory") or row.get("traj") or []
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except json.JSONDecodeError:
            return None
    for step in traj:
        if not isinstance(step, dict):
            continue
        role = step.get("role", "assistant")
        content = step.get("content", "") or step.get("thought", "") or ""
        tool_calls = []
        if step.get("action"):
            tool_calls.append({
                "id": step.get("tool_call_id", f"call_{len(messages)}"),
                "name": step.get("action", ""),
                "arguments": step.get("args", {}) or {},
                "arguments_raw": json.dumps(step.get("args", {}) or {}),
            })
        tool_results = []
        if step.get("observation"):
            tool_results.append({
                "tool_call_id": step.get("tool_call_id", f"call_{len(messages)}"),
                "name": step.get("action", "tool"),
                "content": str(step.get("observation", ""))[:4000],
                "truncated": False,
            })
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_results:
            msg["tool_results"] = tool_results
        messages.append(msg)

    if not messages:
        return None

    return {
        "task": {
            "instruction": row.get("problem_statement") or row.get("issue") or row.get("text", ""),
            "language": "python",
            "task_type": "bug_fix",
            "difficulty": "unlabelled",
            "constraints": [],
            "acceptance_criteria": [],
            "outcome": "unlabelled",
            "task_id": row.get("instance_id") or row.get("task_id", ""),
        },
        "messages": messages,
        "model": row.get("model", "unknown"),
        "started_at": "",
        "verification": {"status": "skipped", "method": "external_dataset"},
        "generator": {
            "source_id": "external_swe_agent",
            "license": "MIT",
            "license_verified": True,
            "redistributable": True,
            "trainable": True,
            "collected_by": "imported",
            "prompt_template": "external",
            "teacher": {"model": row.get("model", "unknown"), "revision": "unknown", "license": "MIT", "sampling": {}},
        },
    }


def convert_openhands(row: dict) -> dict | None:
    """Convert an OpenHands trajectory row to mente format."""
    # OpenHands stores messages as a list of dicts with role/content
    messages_raw = row.get("messages") or row.get("trajectory") or []
    if isinstance(messages_raw, str):
        try:
            messages_raw = json.loads(messages_raw)
        except json.JSONDecodeError:
            return None
    messages = []
    for m in messages_raw:
        if not isinstance(m, dict):
            continue
        msg = {"role": m.get("role", "assistant"), "content": m.get("content", "") or ""}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_results") or m.get("tool_call_results"):
            msg["tool_results"] = m.get("tool_results") or m.get("tool_call_results")
        messages.append(msg)
    if not messages:
        return None
    return {
        "task": {
            "instruction": row.get("problem_statement") or row.get("issue", ""),
            "language": "python",
            "task_type": "bug_fix",
            "difficulty": "unlabelled",
            "constraints": [],
            "acceptance_criteria": [],
            "outcome": "unlabelled",
            "task_id": row.get("instance_id") or row.get("task_id", ""),
        },
        "messages": messages,
        "model": row.get("model", "unknown"),
        "started_at": "",
        "verification": {"status": "skipped", "method": "external_dataset"},
        "generator": {
            "source_id": "external_openhands",
            "license": "MIT",
            "license_verified": True,
            "redistributable": True,
            "trainable": True,
            "collected_by": "imported",
            "prompt_template": "external",
            "teacher": {"model": row.get("model", "unknown"), "revision": "unknown", "license": "MIT", "sampling": {}},
        },
    }


CONVERTERS = {
    "swe_agent": convert_swe_agent,
    "openhands": convert_openhands,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/external")
    ap.add_argument("--out", default="data/external/prepared.jsonl")
    args = ap.parse_args()

    inp = REPO / args.inp
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as f_out:
        for src_file in sorted(inp.glob("*.jsonl")):
            tag = src_file.stem
            converter = CONVERTERS.get(tag)
            if converter is None:
                print(f"  skip {src_file.name} (no converter)")
                continue
            n = 0
            for line in src_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = converter(row)
                if rec is not None:
                    # use GenericAdapter to normalize, then build_record
                    from src.collect.normalizer import GenericAdapter, build_record, RawTrajectory
                    import uuid
                    common = GenericAdapter.to_common(rec["messages"])
                    if not common:
                        continue
                    raw = RawTrajectory(
                        task=rec["task"],
                        messages=common,
                        tool_schemas=[],
                        harness="external",
                        model_id=rec.get("model", ""),
                        started_at=rec.get("started_at", ""),
                        verification=rec.get("verification", {}),
                        generator=rec.get("generator", {}),
                    )
                    record, problems = build_record(raw, f"ext_{tag}_{total:06d}")
                    if record is not None:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total += 1
                        n += 1
            print(f"  {tag}: {n} converted -> {out}")

    print(f"\n  total: {total} trajectories -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())