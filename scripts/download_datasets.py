"""Download free trajectory datasets for training MenteE Code 3B.

    python scripts/download_datasets.py --out data/external --max-rows 200

Downloads from HuggingFace and saves a small subset locally.
For the full run on vast.ai, remove --max-rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SOURCES = [
    ("nebius/SWE-agent-trajectories", "swe_agent"),
    ("nebius/SWE-rebench-openhands-trajectories", "openhands"),
    ("R2E-Gym/R2EGym-SFT-Trajectories", "r2e_gym"),
    ("nvidia/SWE-Zero-openhands-trajectories", "nvidia_zero"),
]


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

    for repo, tag in SOURCES:
        print(f"  downloading {repo} (max {args.max_rows} rows)...")
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception as e:
            # try without split
            try:
                ds = load_dataset(repo, streaming=True)
                ds = ds["train"] if "train" in ds else list(ds.values())[0]
            except Exception as e2:
                print(f"    ! failed: {e2}")
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
        # print column names for the converter
        if rows:
            print(f"    columns: {list(rows[0].keys())}")


if __name__ == "__main__":
    raise SystemExit(main())