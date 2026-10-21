"""Generate a small synthetic replay file so the collector can be proven end to end.

    python scripts/make_sample_replay.py [--out data/raw/replay/sample.jsonl] [--n 12]

The generated file is in the *common* message shape that ``ReplaySource`` reads, so
it exercises the same validation, append and integrity path as a live run — without
needing a GPU, a teacher model or a running agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_collector_loop import synthetic_trajectory  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/replay/sample.jsonl")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(args.n):
            raw = synthetic_trajectory(i)
            fh.write(
                json.dumps(
                    {
                        "task": raw.task,
                        "messages": [
                            {
                                "role": m.role,
                                "content": m.content,
                                "tool_calls": m.tool_calls,
                                "tool_results": m.tool_results,
                            }
                            for m in raw.messages
                        ],
                        "tool_schemas": [
                            {
                                "name": "read_file",
                                "description": "Read a file from the repository.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            },
                            {
                                "name": "apply_patch",
                                "description": "Apply a patch to a file.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            },
                        ],
                        "started_at": "2026-10-21T10:00:00Z",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {out} ({out.stat().st_size} bytes, {args.n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
