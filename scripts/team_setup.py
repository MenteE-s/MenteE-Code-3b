"""One-command setup for a new collector (used by scripts/setup_collector.ps1).

    python scripts/team_setup.py --collector syab --key sk-or-... --shard 2

Does everything a team member needs before their first run:
  * ensures .venv312 exists (creates + installs requirements if missing)
  * scaffolds their collector folder + config via `src.collect init`
  * writes their API key to datasets/collectors/<name>/.env
  * (optionally) prepares their task shard into data/tasks/prepared-<name>.jsonl
  * points their config's tasks_path at that prepared file
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv312" / "Scripts" / "python.exe"


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(REPO), **kw)


def ensure_venv():
    if PY.exists():
        return
    print("  creating .venv312 (one-time, ~1 minute)...")
    subprocess.run([sys.executable, "-m", "venv", str(REPO / ".venv312")], check=True)
    subprocess.run([str(PY), "-m", "pip", "install", "-q", "-r", "requirements-collect.txt"], cwd=str(REPO), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector", required=True)
    ap.add_argument("--key", required=True, help="OpenRouter API key")
    ap.add_argument("--shard", type=int, default=None, help="which shard (1-based) this collector owns")
    ap.add_argument("--collectors", type=int, default=4, help="total number of collectors/shards")
    ap.add_argument("--base-config", default="configs/collect_config.agent.yaml")
    ap.add_argument("--shard-file", default=None, help="explicit shard path (overrides --shard)")
    args = ap.parse_args()

    cid = args.collector.strip().lower()
    ensure_venv()

    # 1. scaffold
    run([str(PY), "-m", "src.collect", "init", "--collector", cid, "--config", args.base_config], check=True)

    # 2. write key
    env_path = REPO / "datasets" / "collectors" / cid / ".env"
    env_path.write_text(f"OPENROUTER_API_KEY={args.key}\n", encoding="utf-8")
    print(f"  wrote key to {env_path}")

    # 3. prepare tasks (no API calls)
    shard_path = None
    if args.shard_file:
        shard_path = Path(args.shard_file)
    elif args.shard is not None:
        candidate = REPO / "data" / "tasks" / "shards" / f"shard-{args.shard}-of-{args.collectors}.jsonl"
        if not candidate.exists():
            candidate = REPO / "data" / "tasks" / f"shard-{args.shard}-of-{args.collectors}.jsonl"
        shard_path = candidate
    prepared = None
    if shard_path and shard_path.exists():
        prepared = REPO / "data" / "tasks" / f"prepared-{cid}.jsonl"
        run([str(PY), "scripts/prepare_tasks.py", "--in", str(shard_path), "--out", str(prepared)], check=True)
    elif shard_path:
        print(f"  ! shard file not found: {shard_path}")
        print("    place your shard at that path (or pass --shard-file) and re-run prepare_tasks.py manually")

    # 4. point config at prepared file
    import yaml
    cfg_path = REPO / "configs" / "collectors" / f"{cid}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if prepared is not None:
        cfg["source"]["tasks_path"] = str(prepared.relative_to(REPO)).replace("\\", "/")
    cfg["collector"]["id"] = cid
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"  config updated: {cfg_path}")

    print()
    print("  SETUP COMPLETE")
    print(f"  Run:  powershell -File scripts\\collect_until_limit.ps1 -Collector {cid}")
    print(f"  or:   {PY} -m src.collect run --config configs/collectors/{cid}.yaml --collector {cid}")


if __name__ == "__main__":
    raise SystemExit(main())