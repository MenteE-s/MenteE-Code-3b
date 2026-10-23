"""Live smoke test: does a free OpenRouter model emit valid tool calls?

Run:  python scripts/test_openrouter_tool.py [model_id]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = "https://openrouter.ai/api/v1"


def key() -> str:
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env
    from pathlib import Path

    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no OPENROUTER_API_KEY (env or .env)")


def call(model: str, tools: list[dict], messages: list[dict]) -> dict:
    body = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "max_tokens": 600}
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://menteeai.org",
            "X-Title": "MenteE Code collector smoke test",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "cohere/north-mini-code:free"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the repository and return its contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to repo root."},
                        "start_line": {"type": "integer", "description": "First line to read."},
                        "end_line": {"type": "integer", "description": "Last line to read."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a unified diff patch to the repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "diff": {"type": "string"},
                    },
                    "required": ["path", "diff"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the test suite and return the results.",
                "parameters": {
                    "type": "object",
                    "properties": {"test_file": {"type": "string"}},
                    "required": [],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding agent. Use the provided tools to investigate and fix the "
                "issue. Read the relevant source before editing. Verify with run_tests when done."
            ),
        },
        {
            "role": "user",
            "content": (
                "In this Python repository, parse_window() raises IndexError when the token "
                "list is empty. The failing test is tests/test_parser.py::test_empty_window. "
                "Fix it. Start by reading src/parser.py."
            ),
        },
    ]

    print(f"model: {model}")
    try:
        resp = call(model, tools, messages)
    except urllib.error.HTTPError as exc:
        print(f"  HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}")
        return 1

    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp.get("usage", {})
    tcs = msg.get("tool_calls") or []

    print(f"  finish_reason : {choice.get('finish_reason')}")
    print(f"  content       : {len(msg.get('content') or '')} chars")
    print(f"  tool_calls    : {len(tcs)}")
    print(f"  usage         : in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
          f"cached={usage.get('prompt_tokens_details', {}).get('cached_tokens')}")
    if tcs:
        tc = tcs[0]
        print(f"  first call    : id={tc.get('id')}  type={tc.get('type')}")
        fn = tc.get("function", {})
        print(f"    name        : {fn.get('name')}")
        raw = fn.get("arguments")
        print(f"    args_raw    : {raw[:160] if isinstance(raw, str) else raw}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            print(f"    parses      : yes -> {parsed}")
        except Exception as exc:  # noqa: BLE001
            print(f"    parses      : NO ({exc})")
            return 1
        print("\n  VERDICT: tool calling works")
        return 0
    print("\n  VERDICT: no tool call emitted — model replied in text only")
    print("  " + (msg.get("content") or "")[:300].replace("\n", "\n  "))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
