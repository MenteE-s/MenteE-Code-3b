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
from .config import load_config, save_config
from .integrity import IntegritySchedule, verify
from .normalizer import build_record
from .report import Reporter
from .source import HarnessSource, ReplaySource, load_tasks
from .state import Sidecar
from .validate import DeepVerifier, TierAValidator, load_schema, make_validator

__all__ = ["main"]


def _load_env_files(collector: str = "") -> None:
    """Fill os.environ from .env files without overriding real environment keys.

    Searched in order (first hit wins, none override the actual environment):
      1. datasets/collectors/<collector>/.env   (collector-specific keys)
      2. repo-root .env
    """
    root = Path(__file__).resolve().parents[2]
    candidates = []
    if collector:
        candidates.append(root / "datasets" / "collectors" / collector / ".env")
    candidates.append(root / ".env")
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# --------------------------------------------------------------------------- helpers
def _seq(sidecar: Sidecar) -> int:
    value = int(sidecar.data.get("next_seq", 0) or 0)
    sidecar.data["next_seq"] = value + 1
    return value


def _record_id(sidecar: Sidecar, collector_id: str = "") -> str:
    stamp = time.strftime("%Y-%m-%d")
    slug = f"{_slug(collector_id)}_" if _slug(collector_id) else ""
    return f"traj_{slug}{stamp}_{_seq(sidecar):06d}"


def _slug(value: str) -> str:
    """Lowercase slug, so 'Syed Syab' and 'syab' are the same collector."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in (value or "").lower())
    return cleaned.strip("-_")


def _heal_duplicate_ids(path: Path, sidecar: Sidecar) -> int:
    """One-time self-heal for a train.jsonl that contains duplicate ids.

    This happens when sidecar state was lost (e.g. a recovered row was appended
    without persisting the sidecar). The integrity checker then fails on every
    run and the collector can never make progress. The heal KEEPS every row and
    only renumbers the colliding ids, so no data is lost. Idempotent: returns 0
    when the file is already clean. Only rewrites when corruption is detected.
    """
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    if not rows:
        return 0
    seen: set[str] = set()
    dupes = []
    for r in rows:
        tid = str(r.get("id") or "")
        if tid in seen:
            dupes.append(r)
        seen.add(tid)
    if not dupes:
        return 0
    max_seq = 0
    for r in rows:
        m = re.search(r"_(\d{6})$", str(r.get("id") or ""))
        if m:
            max_seq = max(max_seq, int(m.group(1)))
    # renumber colliding rows
    seen2: set[str] = set()
    for r in rows:
        tid = str(r.get("id") or "")
        if tid in seen2:
            max_seq += 1
            new_id = re.sub(r"_\d{6}$", f"_{max_seq:06d}", tid)
            if new_id == tid:
                new_id = f"{tid}_{max_seq:06d}"
            print(f"  ! healing duplicate id {tid} -> {new_id}", file=sys.stderr)
            r["id"] = new_id
        seen2.add(str(r.get("id")))
    # atomic rewrite
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    # heal sidecar
    ids = {str(r.get("id")) for r in rows if r.get("id")}
    ids_path = path.with_name(path.name + ".ids")
    ids_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    done: list[str] = []
    for r in rows:
        t = r.get("task") or {}
        tid = t.get("task_id")
        if tid:
            counts[tid] = counts.get(tid, 0) + 1
            if tid not in done:
                done.append(tid)
    sidecar.data["next_seq"] = max(int(sidecar.data.get("next_seq", 0) or 0), max_seq + 1)
    sidecar.data["lines"] = len(rows)
    sidecar.data["bytes"] = path.stat().st_size
    sidecar.data["task_counts"] = counts
    sidecar.data["done_task_ids"] = done
    sidecar.save()
    print(f"  healed {len(dupes)} duplicate id(s); {len(rows)} rows total", file=sys.stderr)
    return len(dupes)
    """Lowercase slug, so 'Syed Syab' and 'syab' are the same collector."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in (value or "").lower())
    return cleaned.strip("-_")


def _load_sidecar(path: Path) -> Sidecar:
    return Sidecar(path).load()


def _build_source(
    cfg: dict[str, Any],
    dry_run: bool,
    skip_task_ids: set[str] | None = None,
    task_counts: dict[str, int] | None = None,
    samples_per_task: int = 1,
):
    from .agent import AgentConfig, AgentRunner, ToolExecutor

    src_cfg = cfg["source"]
    kind = src_cfg.get("kind", "harness")
    generator = dict(src_cfg.get("generator", {}) or {})

    if kind == "agent":
        # live agent loop against any OpenAI-compatible endpoint
        # (OpenRouter, DeepSeek direct, a local vLLM)
        acfg = src_cfg.get("agent", {}) or {}
        teacher = src_cfg.get("teacher", {}) or {}
        api_key = os.environ.get(acfg.get("api_key_env", "OPENROUTER_API_KEY"), "")
        if not api_key:
            raise SystemExit(
                f"source.kind=agent needs {acfg.get('api_key_env', 'OPENROUTER_API_KEY')} in the environment"
            )
        runner = AgentRunner(
            AgentConfig(
                base_url=acfg.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=api_key,
                model=acfg.get("model", "cohere/north-mini-code:free"),
                temperature=float(teacher.get("sampling", {}).get("temperature", 0.7)),
                top_p=float(teacher.get("sampling", {}).get("top_p", 0.95)),
                max_turns=int(acfg.get("max_turns", 25)),
            ),
            ToolExecutor("."),  # replaced per-task inside run()
        )
        skip = skip_task_ids or set()
        counts = task_counts or {}
        tasks = load_tasks(src_cfg["tasks_path"])
        # a task is finished only after it has produced `samples_per_task`
        # accepted trajectories - the same bug can be sampled several times
        # and only the successful attempts are kept
        tasks = [
            t for t in tasks
            if t.task_id not in skip
            or int(counts.get(t.task_id, 0) or 0) < int(samples_per_task)
        ]
        if dry_run:
            tasks = tasks[:1]
        generator.setdefault("harness", "mente-agent")
        generator["teacher"] = {
            "model": acfg.get("model", ""),
            "revision": teacher.get("revision", "unknown"),
            "license": teacher.get("license", "provider terms"),
            "sampling": teacher.get("sampling", {}),
        }
        generator["arguments_raw_source"] = "raw_response"

        class AgentTrajectorySource:
            def __iter__(self):
                for t in tasks:
                    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    raw, meta = runner.run(t.to_record_task() | {"workdir": t.workdir})
                    raw.task.update(meta)
                    raw.started_at = started
                    raw.generator.update(generator)
                    yield raw

        return AgentTrajectorySource()

    if kind == "replay":
        replay_path = src_cfg.get("replay", {}).get("path", "")
        if not replay_path:
            raise SystemExit("source.kind=replay requires source.replay.path")
        return ReplaySource(replay_path, generator=generator)

    tasks = load_tasks(src_cfg["tasks_path"])
    counts = task_counts or {}
    tasks = [
        t for t in tasks
        if t.task_id not in (skip_task_ids or set())
        or int(counts.get(t.task_id, 0) or 0) < int(samples_per_task)
    ]
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

    collector_id = _slug(str(args.collector or (cfg.get("collector", {}) or {}).get("id", "")))
    if args.collector and not out_cfg.get("_explicit_path"):
        root = (cfg.get("collector", {}) or {}).get("output_root", "datasets/collectors")
        out_cfg["path"] = f"{root}/{collector_id}/train.jsonl"
        out_cfg["rejected_path"] = f"{root}/{collector_id}/rejected.jsonl"
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
    sidecar.reconcile_with_file(out_path)
    # self-heal: if a previous run left duplicate ids (e.g. a recovered row
    # was appended without persisting the sidecar), repair them in place so
    # Ctrl+C -> restart always works without manual intervention.
    _heal_duplicate_ids(out_path, sidecar)
    reporter = Reporter(run_id=sidecar.run_id or f"run-{int(time.time())}")
    sidecar.set_run(reporter.run_id, {"config": str(args.config) if args.config else None})

    samples_per_task = int(
        (cfg.get("source", {}).get("agent", {}) or {}).get("samples_per_task", 1)
        or cfg.get("collection", {}).get("samples_per_task", 1)
        or 1
    )
    source = _build_source(
        cfg,
        args.dry_run,
        skip_task_ids=set(sidecar.data.get("done_task_ids", []) or []),
        task_counts=dict(sidecar.data.get("task_counts", {}) or {}),
        samples_per_task=samples_per_task,
    )

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

    provider_limit_hit = False
    provider_error_text = ""
    try:
        for raw in source:
            reporter.generated += 1
            task_id = str(raw.task.get("task_id") or raw.task.get("issue_url") or reporter.generated)
            if (raw.task.get("stop_reason") == "error"):
                err = str(raw.task.get("provider_error") or "")
                if err:
                    # the provider itself failed (rate limit, overload, outage).
                    # Every remaining task would fail the same way, so stop and
                    # let the run script restart later.
                    provider_limit_hit = True
                    provider_error_text = err
                    print(f"  provider error: {err[:200]}", file=sys.stderr)
                    break

            record, problems = build_record(raw, _record_id(sidecar, collector_id))
            if collector_id:
                record["provenance"]["collected_by"] = collector_id
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
            task_id = (record.get("task") or {}).get("task_id")
            if task_id:
                sidecar.add_task(task_id)
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

    if provider_limit_hit:
        print(f"  provider error - stopping (restart later): {provider_error_text[:200]}", file=sys.stderr)
        return 3

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


# --------------------------------------------------------------------------- team
def cmd_init(args: argparse.Namespace) -> int:
    """Scaffold one collector: their own folder, config and .env template."""
    cid = _slug(args.collector)
    if not cid:
        print("  ! --collector is required (e.g. --collector syab)")
        return 1
    cfg = load_config(args.config)
    cfg["collector"] = {"id": cid, "output_root": "datasets/collectors"}
    cfg["output"]["path"] = f"datasets/collectors/{cid}/train.jsonl"
    cfg["output"]["rejected_path"] = f"datasets/collectors/{cid}/rejected.jsonl"

    cfg_path = Path(f"configs/collectors/{cid}.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg_path)

    Path(f"datasets/collectors/{cid}").mkdir(parents=True, exist_ok=True)
    env_path = Path(f"datasets/collectors/{cid}/.env")
    if not env_path.exists():
        env_path.write_text(
            "OPENROUTER_API_KEY=paste-your-key-here\n"
            "DEEPSEEK_API_KEY=paste-your-key-here\n",
            encoding="utf-8",
        )
    print(f"  collector : {cid}")
    print(f"  config    : {cfg_path}")
    print(f"  output    : {cfg['output']['path']}")
    print(f"  env       : {env_path}  (put your key here; .env is gitignored)")
    print(f"\n  run:  python -m src.collect run --config {cfg_path}")
    return 0


def cmd_partition(args: argparse.Namespace) -> int:
    """Split a task file into N disjoint shards so collectors never overlap.

    Deterministic: the same task file and N always produce the same shards, so
    re-running it (or another team member re-running it) agrees.
    """
    import hashlib

    tasks = load_tasks(args.tasks)
    n = max(1, args.collectors)
    shards: list[list[Any]] = [[] for _ in range(n)]
    for t in tasks:
        idx = int(hashlib.sha256(t.task_id.encode("utf-8")).hexdigest()[:8], 16) % n
        shards[idx].append(t)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, shard in enumerate(shards):
        p = out_dir / f"shard-{i + 1}-of-{n}.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for t in shard:
                fh.write(
                    json.dumps(
                        {
                            "task_id": t.task_id,
                            "instruction": t.instruction,
                            "repo": t.repo,
                            "base_commit": t.base_commit,
                            "setup_script": t.setup_script,
                            "test_command": t.test_command,
                            "language": t.language,
                            "task_type": t.task_type,
                            "difficulty": t.difficulty,
                            "workdir": t.workdir,
                            "fail_to_pass": t.fail_to_pass or [],
                            "pass_to_pass": t.pass_to_pass or [],
                            **({"issue_url": t.issue_url} if t.issue_url else {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  {p.name}: {len(shard)} task(s)")
    print(f"\n  {n} disjoint shards written to {out_dir}")
    print("  every task_id is in exactly one shard, so no two collectors overlap.")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from .merge import merge_datasets

    schema_path = Path(args.schema) if args.schema else Path("schemas/mentee-trajectory-v1.schema.json")
    schema = load_schema(schema_path) if schema_path.exists() else None
    validator_obj = make_validator(schema) if schema else None

    inputs = [Path(p) for p in args.inputs]
    inputs = [p for p in inputs if p.exists() and p.stat().st_size > 0]
    dropped = len(args.inputs) - len(inputs)
    if dropped:
        print(f"  ({dropped} empty/missing input(s) skipped)")
    if not inputs:
        print("  ! nothing to merge")
        return 1

    out_path = Path(args.out)
    report = merge_datasets(inputs, out_path, validator=validator_obj, verbose=True)

    print("\n  merge " + ("OK" if report.ok else "FAILED"))
    print(f"  accepted total : {report.total_accepted}")
    print(f"  duplicate ids  : {report.total_duplicate_id}")
    print(f"  duplicate content: {report.total_duplicate_content}")
    print(f"  schema invalid : {report.total_schema_invalid}")
    print(f"  output         : {report.output_path} ({report.output_lines} rows)")
    for note in report.notes:
        print(f"  ! {note}")

    verified = sum(s.verified for s in report.sources)
    fp = sum(s.failure_path for s in report.sources)
    if report.total_accepted:
        print("\n  quality across collectors:")
        print(f"    verified rows   {verified} ({verified / max(1, report.total_accepted) * 100:.0f}%)")
        print(f"    failure-path    {fp} ({fp / max(1, report.total_accepted) * 100:.0f}%)  <- target >= 30%")
        print("    per collector:")
        for s in report.sources:
            rate = (s.verified / s.accepted * 100) if s.accepted else 0
            print(f"      {Path(s.path).parent.name:<16} kept={s.accepted:<6} verified={rate:.0f}%")

    rep_path = out_path.with_name(out_path.stem + ".merge-report.json")
    rep_path.write_text(
        json.dumps(
            {
                "ok": report.ok,
                "total_accepted": report.total_accepted,
                "duplicate_id": report.total_duplicate_id,
                "duplicate_content": report.total_duplicate_content,
                "schema_invalid": report.total_schema_invalid,
                "output": report.output_path,
                "output_lines": report.output_lines,
                "sources": [vars(s) for s in report.sources],
                "notes": report.notes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  report: {rep_path}")
    return 0 if report.ok else 1



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
    p_run.add_argument("--collector", default=None, help="collector id (e.g. syab) - namespaces ids and output")
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

    p_init = sub.add_parser("init", help="scaffold one collector (their own folder + config)")
    p_init.add_argument("--collector", required=True, help="collector id, e.g. syab")
    p_init.add_argument("--config", default=None, help="base config to copy from")
    p_init.set_defaults(func=cmd_init)

    p_part = sub.add_parser("partition", help="split a task file into N disjoint shards")
    p_part.add_argument("--tasks", required=True)
    p_part.add_argument("--collectors", type=int, required=True)
    p_part.add_argument("--out", default="data/tasks/shards")
    p_part.set_defaults(func=cmd_partition)

    p_merge = sub.add_parser("merge", help="merge several collectors' datasets into one")
    p_merge.add_argument("--inputs", nargs="+", required=True, help="collector train.jsonl paths")
    p_merge.add_argument("--out", required=True)
    p_merge.add_argument("--schema", default=None)
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    # keys come from the environment; .env files top it up (never override)
    _load_env_files(getattr(args, "collector", "") or "")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
