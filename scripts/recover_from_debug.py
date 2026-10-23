"""Recover a trajectory from a debug run file into the dataset.

    python scripts/recover_from_debug.py datasets/collectors/syab/debug-traj_syab_2026-10-23_000001.json

Rebuilds the record from the saved messages, re-validates it, and appends it with
a fresh id. Used when a run was accepted but lost to an id clash, or when a
rejection turned out to be a validator bug rather than a data problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.collect.appender import JsonlAppender  # noqa: E402
from src.collect.cli import _record_id, _slug  # noqa: E402
from src.collect.integrity import verify  # noqa: E402
from src.collect.normalizer import GenericAdapter, build_record  # noqa: E402
from src.collect.state import Sidecar  # noqa: E402
from src.collect.validate import load_schema, make_validator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("debug_file")
    ap.add_argument("--collector", default="syab")
    ap.add_argument("--verify-status", default=None, help="override verification status (pass/fail/skipped)")
    ap.add_argument("--verify-method", default=None)
    ap.add_argument("--drop-last-failed-run_tests", action="store_true",
                    help="when the debug run ended mid-testing, drop a trailing failing run_tests result")
    args = ap.parse_args()

    data = json.loads(Path(args.debug_file).read_text(encoding="utf-8"))
    messages = list(data.get("messages") or [])

    # Normalise: some models emit the same call twice in one turn. Keep the first.
    seen_calls: set[tuple[str, str]] = set()
    seen_results: set[str] = set()
    clean: list[dict] = []
    for m in messages:
        calls = m.get("tool_calls") or []
        kept: list[dict] = []
        for c in calls:
            fn = c.get("function", c)
            cid = str(c.get("id") or "")
            key = ("id", cid) if cid else ("args", fn.get("name"), str(fn.get("arguments", "{}")))
            if key in seen_calls:
                continue
            seen_calls.add(key)
            kept.append(c)
        if calls:
            m["tool_calls"] = kept
            # the debug dump is written after build_record, whose merge step
            # attaches results to assistant messages in place, so each result
            # is carried twice (on the assistant and in the standalone
            # tool_result message). Treat the standalone messages as the
            # source of truth and strip the attached copy.
            m["tool_results"] = []

        res = m.get("tool_results") or []
        if res:
            unique_res = []
            for r in res:
                rid2 = r.get("tool_call_id")
                if rid2 in seen_results:
                    continue
                seen_results.add(rid2)
                unique_res.append(r)
            m["tool_results"] = unique_res
            if not unique_res:
                continue
        clean.append(m)
    messages = clean

    # recover outcome evidence from the run itself: the last run_tests result
    last_test = None
    for m in reversed(messages):
        if m.get("role") != "tool_result":
            continue
        for r in m.get("tool_results", []) or []:
            if r.get("name") == "run_tests":
                last_test = r
                break
        if last_test is not None:
            break

    if args.drop_last_failed_run_tests and last_test is not None and not last_test["content"].startswith("tests passed"):
        # remove the trailing failed run_tests exchange (assistant + result)
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "tool_result" and any(
                r.get("name") == "run_tests" for r in m.get("tool_results", [])
            ):
                messages = messages[:i]
                break

    common = GenericAdapter.to_common(messages)
    if not common:
        print("  !! no messages recovered")
        return 1

    sidecar = Sidecar(REPO / f"datasets/collectors/{_slug(args.collector)}" / "train.jsonl").load()
    sidecar.reconcile_with_file(REPO / f"datasets/collectors/{_slug(args.collector)}" / "train.jsonl")
    sidecar.open()
    rid = _record_id(sidecar, args.collector)

    from src.collect.normalizer import RawTrajectory  # noqa: E402

    meta = data.get("meta") or {}
    task = data.get("task") or {}

    generator = {
        "source_id": "mente_synth_v1",
        "source_url": "https://openrouter.ai/api/v1",
        "license": "NVIDIA-Open-Model-License",
        "license_verified": True,   # nvidia.com: commercially usable, derivative models permitted
        "redistributable": True,
        "trainable": True,
        "collected_by": _slug(args.collector),
        "prompt_template": "mente_agent_v1",
        "teacher": {
            "model": meta.get("model", "unknown"),
            "revision": "free",
            "license": "NVIDIA-Open-Model-License",
            "sampling": {"temperature": 0.7, "top_p": 0.95},
        },
    }

    raw = RawTrajectory(
        task=task,
        messages=common,
        tool_schemas=[],
        harness="mente-agent",
        model_id=meta.get("model", ""),
        started_at=meta.get("started_at", ""),
        verification=data.get("verification", {}),
        generator=generator,
    )
    record, problems = build_record(raw, rid)
    if record is None:
        print(f"  !! could not build a record: {problems}")
        return 1

    record["provenance"]["collected_by"] = _slug(args.collector)
    if args.verify_status:
        record["verification"]["status"] = args.verify_status
    if args.verify_method:
        record["verification"]["method"] = args.verify_method
    record["quality"]["verified"] = record["verification"]["status"] == "pass"

    schema = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
    validator = make_validator(schema)
    errs = list(validator.iter_errors(record)) if validator else []
    if errs:
        print(f"  !! record fails schema: {errs[0].message}")
        return 1

    from src.collect.validate import TierAValidator  # noqa: E402

    tier_a = TierAValidator(validator=validator)
    verdict = tier_a.validate(record)
    print(f"  validation: {'PASS' if verdict.ok else 'REJECT'}")
    for reason in verdict.reasons:
        print(f"    - {reason}")
    if not verdict.ok:
        return 1

    out = REPO / f"datasets/collectors/{_slug(args.collector)}" / "train.jsonl"
    with JsonlAppender(out, fsync_every=1) as app:
        app.append(record)
    sidecar.add_id(record["id"])
    sidecar.add_id(record["id"])
    sidecar.add_task((record.get("task") or {}).get("task_id", ""))
    sidecar.close()
    print(f"  recovered {record['id']} -> {out}")
    report = verify(out, deep=True, schema=schema, validator=validator)
    print(f"  file check: {report.summary()}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
