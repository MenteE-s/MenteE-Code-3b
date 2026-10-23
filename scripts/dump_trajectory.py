"""Dump the messages of a trajectory or a debug run file, for debugging failed runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    rec = data.get("record")
    if rec is not None:
        print("REJECTION reasons:", data.get("reasons"))

    if "messages" in data:
        messages = data["messages"]
    elif "trajectory" in data:
        messages = data["trajectory"]["turns"]
    else:
        messages = []

    print("messages:", len(messages))
    for t in messages:
        role = t.get("role")
        if t.get("tool_calls"):
            for c in t["tool_calls"]:
                fn = c.get("function", c)
                name = fn.get("name")
                raw = fn.get("arguments", "{}")
                print(f"  ASSISTANT -> {name}({str(raw)[:110]})")
                args = c.get("arguments")
                if name == "apply_patch" and isinstance(args, dict):
                    diff = args.get("diff", "")
                    print(f"        diff[{len(diff)} chars] head: {diff[:140]!r}")
        elif role == "tool_result":
            for r in t.get("tool_results", []) or []:
                print(f"  RESULT    : {r['content'][:130]!r}")
            if not t.get("tool_results") and "content" in t:
                print(f"  RESULT    : {t['content'][:130]!r}")
        elif role == "assistant":
            print(f"  ASSISTANT : {(t.get('content') or '')[:130]!r}")
        else:
            print(f"  {str(role).upper():<9}: {(t.get('content') or '')[:110]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
