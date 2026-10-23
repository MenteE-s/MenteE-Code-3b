"""Download a small slice of free trajectory datasets for local smoke testing.

    python scripts/download_datasets.py --out data/external --max-rows 200

Downloads from HuggingFace (SWE-agent trajectories + OpenHands trajectories)
and saves a small subset locally. For the full run on vast.ai, remove --max-rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external")
    ap.add_argument("--max-rows", type=int, default=200, help="max rows per source (smoke test)")
    args = ap.parse_args()

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ! install: pip install datasets")
        raise SystemExit(1)

    sources = [
        ("SWE-agent/SWE-agent-trajectories", None, "swe_agent"),
        ("All-Hands-AI/OpenHands-trajectories", None, "openhands"),
    ]

    for repo, subset, tag in sources:
        print(f"  downloading {repo} (max {args.max_rows} rows)...")
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception as e:
            print(f"    ! failed: {e}")
            continue
        rows = []
        for i, row in enumerate(ds):
            if i >= args.max_rows:
                break
            rows.append(row)
        dest = out / f"{tag}.jsonl"
        with dest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"    saved {len(rows)} rows -> {dest}")


if __name__ == "__main__":
    raise SystemExit(main())