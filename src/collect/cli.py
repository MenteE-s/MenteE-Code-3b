"""Command line interface for the collector.

::

    python -m src.collect run     --config configs/collect_config.yaml [--limit N] [--dry-run]
    python -m src.collect verify  --path data/processed/v1.0.0/train.jsonl [--deep]
    python -m src.collect status  --path data/processed/v1.0.0/train.jsonl
    python -m src.collect tail    --path data/processed/v1.0.0/train.jsonl -n 5
    python -m src.collect report  --path data/processed/v1.0.0/train.jsonl

The ``run`` loop is deliberately simple: generate one trajectory, validate it,
append it, repeat. There is no batching and no file rewriting — the dataset file
is only ever appended to.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from .appender import AppendError, JsonlAppender
from .budget import Budget
from .config import load_config
from .integrity import IntegritySchedule, verify
from .normalizer import build_record
from .report import Reporter
from .source import HarnessSource, ReplaySource, load_tasks
from .state import Sidecar
from .validate import DeepVerifier, TierAValidator, load_schema, make_validator

__all__ = ["main"]


# --------------------------------------------------------------------------- helpers
def _seq(sidecar: Sidecar) -> int:
    value = int(sidecar.data.get("next_seq", 0) or 0)
    sidecar.data["next_seq"] = value + 1
    return value


def _record_id(sidecar: Sidecar) -> str:
    stamp = time.strftime("%Y-%m-%d")
    return f"traj_{stamp}_{_seq(sidecar):06d}"


def _load_sidecar(path: Path) -> Sidecar:
    return Sidecar(path).load()


def _build_source(cfg: dict[str, Any], dry_run: bool):
    src_cfg = cfg["source"]
    kind = src_cfg.get("kind", "harness")
    generator = dict(src_cfg.get("generator", {}) or {})

    if kind == "replay":
        replay_path = src_cfg.get("replay", {}).get("path", "")
        if not replay_path:
            raise SystemExit("source.kind=replay requires source.replay.path")
        return ReplaySource(replay_path, generator=generator)

    tasks = load_tasks(src_cfg["tasks_path"])
    if dry_run:
        tasks = tasks[:1]

    oc = src_cfg.get("opencode", {}) or {}
    teacher = src_cfg.get("teacher", {}) or {}
    generator.setdefault("harness", src_cfg.get("harness", "opencode"))
    generator["teacher"] = {
        "model": teacher.get("model", ""),
        "revision": teacher.get("revision", "main"),
        "license": teacher.get("license", "unknown"),
        "sampling": teacher.get("sampling", {}),
    }
    password = ""
    env_name = oc.get("auth_password_env", "OPENCODE_SERVER_PASSWORD")
    if env_name:
        password = os.environ.get(env_name, "")
    return HarnessSource(
        tasks=tasks,
        server_url=oc.get("server_url", "http://127.0.0.1:4096"),
        auth_user=oc.get("auth_user", "opencode"),
        auth_password=password,
        model=oc.get("model", "") or teacher.get("model", ""),
        agent=oc.get("agent", ""),
        timeout_s=int(oc.get("timeout_s", 1800)),
        generator=generator,
    )


def _shuffle_tools(cfg: dict[str, Any], tools: list[dict[str, Any]], task_id: str):
    """Apply the tool-schema shuffler to a tool set, if enabled."""
    scfg = cfg["source"].get("tool_schema_shuffle", {}) or {}
    if not scfg.get("enabled") or not tools:
        return tools, None
    from .schema_tools import ToolShuffler

    shuffler = ToolShuffler(
        rename_probability=float(scfg.get("rename_probability", 0.8)),
        rename_params_probability=float(scfg.get("rename_params_probability", 0.7)),
        drop_probability=float(scfg.get("drop_probability", 0.25)),
        add_probability=float(scfg.get("add_probability", 0.35)),
    )
    rng = random.Random(ToolShuffler.seed_for(task_id, "tools"))
    if rng.random() > float(scfg.get("probability", 0.35)):
        return tools, None
    result = shuffler.shuffle(tools, ToolShuffler.seed_for(task_id, "tools"))
    return result.tools, result


# --------------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_cfg = cfg["output"]
    out_path = Path(out_cfg["path"])
    rejected_path = Path(out_cfg["rejected_path"])

    schema_path = Path(cfg["validation"]["schema_path"])
    schema = load_schema(schema_path) if schema_path.exists() else None
    validator_obj = make_validator(schema) if schema else None
    if schema and validator_obj is None:
        print("  ! jsonschema is not installed — schema validation is disabled", file=sys.stderr)
    elif not schema:
        print(f"  ! schema not found at {schema_path} — schema validation is disabled", file=sys.stderr)

    tier_a = TierAValidator(
        validator=validator_obj,
        min_chars=int(cfg["validation"].get("min_chars", 400)),
        max_chars=int(cfg["validation"].get("max_chars", 400000)),
        max_identical_tool_calls=int(cfg["validation"].get("max_identical_tool_calls", 3)),
    )
    deep = DeepVerifier(timeout_s=int(cfg["validation"].get("tier_b_timeout_s", 900)))
    sample_rate = float(cfg["validation"].get("tier_b_sample_rate", 0.02))
    require_verified = bool(cfg["validation"].get("require_verified", False))

    integ = cfg["integrity"]
    schedule = IntegritySchedule(
        first_check_at=int(integ.get("first_check_at", 10)),
        gaps=integ.get("gaps", [30, 45, 60, 90]),
        steady_every=int(integ.get("steady_every", 200)),
        jitter_ratio=float(integ.get("jitter_ratio", 0.30)),
        full_rescan_every=int(integ.get("full_rescan_every", 5)),
    )

    budget = Budget(
        max_rows=args.limit if args.limit else cfg["budget"].get("max_rows"),
        max_minutes=cfg["budget"].get("max_minutes"),
        max_generation_tokens=cfg["budget"].get("max_generation_tokens"),
    )

    sidecar = _load_sidecar(out_path)
    reporter = Reporter(run_id=sidecar.run_id or f"run-{int(time.time())}")
    sidecar.set_run(reporter.run_id, {"config": str(args.config) if args.config else None})

    source = _build_source(cfg, args.dry_run)

    appender = JsonlAppender(out_path, fsync_every=int(out_cfg.get("fsync_every", 10)))
    rejected_appender = JsonlAppender(rejected_path, fsync_every=10)
    try:
        appender.open(resume=bool(out_cfg.get("resume", True)))
        rejected_appender.open(resume=True)
        sidecar.open()
    except AppendError as exc:
        print(f"  ! {exc}", file=sys.stderr)
        return 2

    if appender.healed_bytes:
        reporter.note(
            "heal",
            f"discarded {appender.healed_bytes} bytes of a partial trailing record",
        )
        print(f"  healed {appender.healed_bytes} bytes of a partial trailing record")

    print(f"  appending to   {out_path}")
    print(f"  starting at    line {appender.lines} ({appender.bytes} bytes)")
    print(f"  next check at  {schedule.next_threshold()} rows")
    print("")

    try:
        for raw in source:
            reporter.generated += 1
            task_id = str(raw.task.get("task_id") or raw.task.get("issue_url") or reporter.generated)

            record, problems = build_record(raw, _record_id(sidecar))
            if record is None:
                reporter.skip("normalize_failed")
                reporter.note("normalize_failed", "; ".join(problems)[:300])
                continue

            if args.dry_run:
                verdict = tier_a.validate(record)
                status = "PASS" if verdict.ok else "REJECT"
                print(f"  [{status}] {record['id']}  ({verdict.summary() if not verdict.ok else 'ok'})")
                for reason in verdict.reasons:
                    print(f"        - {reason}")
                if not verdict.ok:
                    continue
                print(json.dumps(record, indent=2, ensure_ascii=False)[:4000])
                break

            verdict = tier_a.validate(record)
            if not verdict.ok:
                rejected_appender.append(
                    {
                        "id": record["id"],
                        "reasons": verdict.reasons,
                        "checks": {c.name: c.ok for c in verdict.checks},
                        "record": record,
                    }
                )
                reporter.reject(verdict.reasons)
                continue

            if sidecar.has_id(record["id"]):
                reporter.skip("duplicate_id")
                continue

            # --- Tier B: deep verification on a sample, BEFORE appending -------
            if sample_rate > 0 and random.random() < sample_rate:
                reporter.deep_checked += 1
                evidence = deep.verify(record, raw.task.get("workdir"))
                if evidence.get("status") == "fail":
                    record["verification"].update(
                        {
                            "status": "fail",
                            "verifier": "deep-verifier@1.0.0",
                        }
                    )
                    record["quality"]["verified"] = False
                    verdict = tier_a.validate(record)
                    if not verdict.ok:
                        rejected_appender.append(
                            {
                                "id": record["id"],
                                "reasons": verdict.reasons + [f"tier_b: {evidence.get('reason','')}"],
                                "record": record,
                            }
                        )
                        reporter.reject([f"tier_b: {evidence.get('reason', 'failed')}"])
                        continue
                elif evidence.get("status") == "pass":
                    record["verification"].update(evidence)
                    record["quality"]["verified"] = True

            if require_verified and record["verification"].get("status") != "pass":
                rejected_appender.append(
                    {
                        "id": record["id"],
                        "reasons": ["require_verified: row is not verified"],
                        "record": record,
                    }
                )
                reporter.reject(["require_verified: row is not verified"])
                continue

            appender.append(record)
            sidecar.add_id(record["id"])
            reporter.accept(record)
            budget.consume(
                rows=1,
                tokens=int((record.get("trajectory", {}).get("total_tokens", {}) or {}).get("output", 0) or 0),
            )

            # --- integrity checkpoint on the widening schedule ----------------
            if integ.get("enabled", True) and schedule.due(appender.lines):
                reporter.integrity_checks += 1
                report = verify(
                    out_path,
                    believed_lines=appender.lines,
                    believed_bytes=appender.bytes,
                    deep=schedule.is_deep(),
                    schema=schema,
                    validator=validator_obj,
                )
                if not report.ok:
                    reporter.integrity_failures += 1
                    reporter.note("integrity", report.summary())
                    print(f"  ! INTEGRITY FAILURE\n    {report.summary()}", file=sys.stderr)
                    if integ.get("on_failure", "abort") == "abort":
                        print("  aborting: the file on disk does not match what we wrote", file=sys.stderr)
                        break
                else:
                    deep_tag = " (full re-parse)" if schedule.is_deep() else ""
                    print(f"  [ok] integrity ok at {appender.lines} rows{deep_tag}")
                schedule.mark_checked()
                appender.checkpoint()
                sidecar.checkpoint(
                    lines=appender.lines,
                    bytes_written=appender.bytes,
                    last_id=record["id"],
                    rejections=dict(reporter.rejections),
                )

            exhausted = budget.exhausted()
            if exhausted:
                reporter.note("budget", f"stopped: {exhausted}")
                print(f"  budget reached ({exhausted}) — stopping")
                break
    except KeyboardInterrupt:
        reporter.note("interrupt", "interrupted by user")
        print("\n  interrupted — checkpointing")
    finally:
        appender.checkpoint()
        sidecar.checkpoint(
            lines=appender.lines,
            bytes_written=appender.bytes,
            rejections=dict(reporter.rejections),
        )
        appender.close()
        rejected_appender.close()
        sidecar.close()

    # --- final verification --------------------------------------------------
    final = verify(
        out_path,
        believed_lines=appender.lines,
        believed_bytes=appender.bytes,
        deep=True,
        schema=schema,
        validator=validator_obj,
    )
    print(f"\n  final integrity: {final.summary()}")
    reporter.integrity_checks += 1
    if not final.ok:
        reporter.integrity_failures += 1

    report_path = out_path.with_name(out_path.stem + ".report.json")
    reporter.write(report_path)
    print(reporter.render())
    print(f"\n  report written to {report_path}")
    return 0 if final.ok else 1


# --------------------------------------------------------------------------- others
def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.path)
    sidecar = _load_sidecar(path)
    schema_path = Path(args.schema) if args.schema else Path(
        "schemas/mentee-trajectory-v1.schema.json"
    )
    schema = load_schema(schema_path) if schema_path.exists() else None
    validator_obj = make_validator(schema) if schema else None
    report = verify(
        path,
        believed_lines=sidecar.lines or None,
        believed_bytes=sidecar.bytes or None,
        deep=not args.shallow,
        schema=schema,
        validator=validator_obj,
    )
    print(report.summary())
    if report.parse_errors:
        for line, err in report.parse_errors[:20]:
            print(f"    line {line}: {err}")
    if report.schema_errors:
        for line, err in report.schema_errors[:20]:
            print(f"    line {line}: {err}")
    return 0 if report.ok else 1


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.path)
    sidecar = _load_sidecar(path)
    data = sidecar.data
    print(f"  path            {path}")
    print(f"  exists          {path.exists()}")
    print(f"  believed lines  {data.get('lines', 0)}")
    print(f"  believed bytes  {data.get('bytes', 0)}")
    print(f"  last id         {data.get('last_id', '-')}")
    print(f"  run id          {data.get('run_id', '-')}")
    print(f"  last checkpoint {data.get('checkpointed_at', '-')}")
    if data.get("rejections"):
        print("  rejections:")
        for name, count in sorted(data["rejections"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<32} {count}")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"  ! {path} does not exist")
        return 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.n :]:
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! unparseable line: {exc}")
            continue
        traj = obj.get("trajectory", {})
        ver = obj.get("verification", {})
        task = obj.get("task", {})
        print(
            f"  {obj.get('id')}  turns={traj.get('n_turns')}  "
            f"status={ver.get('status')}  type={task.get('task_type')}  "
            f"repo={obj.get('environment', {}).get('repo')}"
        )
        print(f"      {task.get('instruction','')[:140]}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.path)
    report_path = path.with_name(path.stem + ".report.json")
    if not report_path.exists():
        print(f"  ! no report at {report_path}")
        return 1
    print(json.dumps(json.loads(report_path.read_text(encoding="utf-8")), indent=2))
    return 0


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252 and will raise on a non-ASCII symbol.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="mentee-collect", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate, validate and append trajectories")
    p_run.add_argument("--config", default=None, help="path to the collector config")
    p_run.add_argument("--limit", type=int, default=None, help="stop after N accepted rows")
    p_run.add_argument("--dry-run", action="store_true", help="generate one row and print it")
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify", help="verify the integrity of a dataset file")
    p_verify.add_argument("--path", required=True)
    p_verify.add_argument("--schema", default=None)
    p_verify.add_argument(
        "--shallow",
        action="store_true",
        help="only check the tail instead of re-parsing every line",
    )
    p_verify.set_defaults(func=cmd_verify)

    p_status = sub.add_parser("status", help="show the collector's sidecar state")
    p_status.add_argument("--path", required=True)
    p_status.set_defaults(func=cmd_status)

    p_tail = sub.add_parser("tail", help="show the last N records")
    p_tail.add_argument("--path", required=True)
    p_tail.add_argument("-n", type=int, default=5)
    p_tail.set_defaults(func=cmd_tail)

    p_report = sub.add_parser("report", help="print the last run report")
    p_report.add_argument("--path", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
