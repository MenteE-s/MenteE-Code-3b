"""One-time repair for a train.jsonl that contains duplicate trajectory ids.

This happens when the sidecar state was lost (e.g. a collector recovered a row
from a debug dump without persisting the sidecar). The integrity checker then
fails on every run and the collector can never make progress.

The repair KEEPS every row (no data is lost) and only renumbers the colliding
ids, so each trajectory gets a unique id. It is idempotent: running it on an
already-clean file does nothing.

    python scripts/repair_duplicate_ids.py --collector syab
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collector", required=True)
    ap.add_argument("--path", default=None, help="explicit train.jsonl path")
    args = ap.parse_args()

    if args.path:
        data_path = Path(args.path)
        if not data_path.is_absolute():
            data_path = REPO / data_path
    else:
        data_path = REPO / "datasets" / "collectors" / args.collector / "train.jsonl"

    if not data_path.is_file():
        print(f"  no file at {data_path}")
        return 0

    lines = data_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    if not rows:
        print("  empty file, nothing to repair")
        return 0

    # highest seq number already in use
    max_seq = 0
    for r in rows:
        tid = str(r.get("id") or "")
        m = re.search(r"_(\d{6})$", tid)
        if m:
            max_seq = max(max_seq, int(m.group(1)))

    seen: set[str] = set()
    repaired = 0
    for r in rows:
        tid = str(r.get("id") or "")
        if tid in seen:
            max_seq += 1
            new_id = re.sub(r"_\d{6}$", f"_{max_seq:06d}", tid)
            # if the id had no trailing seq, append one
            if new_id == tid:
                new_id = f"{tid}_{max_seq:06d}"
            print(f"  repair: {tid}  ->  {new_id}")
            r["id"] = new_id
            repaired += 1
        seen.add(r["id"])

    if repaired == 0:
        print("  no duplicate ids found - file is clean")
        return 0

    # atomic rewrite
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, data_path)
    print(f"  repaired {repaired} id(s); {len(rows)} rows total")

    # heal the sidecar state
    state_path = data_path.with_name(data_path.name + ".state.json")
    ids_path = data_path.with_name(data_path.name + ".ids")
    ids = {str(r.get("id")) for r in rows if r.get("id")}
    ids_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state["next_seq"] = max(int(state.get("next_seq", 0) or 0), max_seq + 1)
    state["lines"] = len(rows)
    state["bytes"] = data_path.stat().st_size
    # rebuild task counts / done list from the repaired file
    counts: dict[str, int] = {}
    done: list[str] = []
    for r in rows:
        t = r.get("task") or {}
        tid = t.get("task_id")
        if tid:
            counts[tid] = counts.get(tid, 0) + 1
            if tid not in done:
                done.append(tid)
    state["task_counts"] = counts
    state["done_task_ids"] = done
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  sidecar state healed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())