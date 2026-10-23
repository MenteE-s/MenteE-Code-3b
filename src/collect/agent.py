"""A minimal agent runtime: the model proposes, this executes.

This is the piece that turns a *task* into a *trajectory*. The collector can
already validate and append; this module is what generates something worth
validating.

Boundary (docs/03-ARCHITECTURE.md section 2)
--------------------------------------------
The model may only **propose**. This runtime **disposes**: it decides whether a
proposed action is allowed, executes it inside the task workdir, and reports what
actually happened. Three rules are enforced here, not in the prompt:

* **workspace scoping** - a proposed path must resolve inside the workdir. Path
  traversal (``../../etc/passwd``) is refused, not clamped.
* **secret hygiene** - files that look like credentials (``.env``, key material,
  ``.git``) are never read. If the model echoed a real key into a tool result, the
  secret scan would reject the row afterwards; refusing to read it is better.
* **resource caps** - bounded output sizes with an explicit ``truncated`` flag
  (silent truncation teaches the model to hallucinate), and hard timeouts on any
  subprocess.

Tool set
--------
Deliberately small, because every tool the model can call is a tool it must learn
to call correctly: ``read_file``, ``list_files``, ``search_code``, ``apply_patch``,
``run_tests``.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalizer import CommonMessage, RawTrajectory

__all__ = ["ToolError", "ToolExecutor", "AgentRunner", "TOOLS"]

_MAX_TOOL_OUTPUT = 9000  # chars; anything longer is clipped and flagged
_MAX_LIST = 400
_READ_LIMIT = 60_000  # bytes read from one file


class ToolError(Exception):
    """A proposed action was refused or failed to execute."""


# --------------------------------------------------------------------------- tools
@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str
    truncated: bool = False
    error: str | None = None


def _clip(text: str, limit: int = _MAX_TOOL_OUTPUT) -> tuple[str, bool]:
    """Head+tail clip, preserving the end (errors live at the end)."""
    if len(text) <= limit:
        return text, False
    head, tail = limit * 2 // 3, limit // 3
    return (
        text[:head] + f"\n\n... [clipped {len(text) - limit} chars] ...\n\n" + text[-tail:],
        True,
    )


_SECRET_FILE = re.compile(
    r"(^|/)(\.env(\..*)?|credentials?\..*|.*_key.*|.*secret.*|id_rsa.*|\.git/)", re.I
)


class ToolExecutor:
    """Executes proposed tool calls inside one task workdir."""

    def __init__(self, workdir: str | Path, test_command: str = "", timeout_s: int = 300):
        self.workdir = Path(workdir).resolve()
        if not self.workdir.exists():
            raise FileNotFoundError(self.workdir)
        self.test_command = test_command or "python -m pytest -q"
        self.timeout_s = timeout_s

    # ---------------------------------------------------------------- safety
    def _safe(self, rel: str) -> Path:
        if not rel or not isinstance(rel, str):
            raise ToolError("path is required")
        rel = rel.strip().replace("\\", "/").lstrip("/")
        candidate = (self.workdir / rel).resolve()
        cand = str(candidate)
        root = str(self.workdir)
        if not (cand == root or cand.startswith(root + os.sep)):
            raise ToolError(f"path escapes the workspace: {rel}")
        if _SECRET_FILE.search("/" + rel):
            raise ToolError(f"refusing to read a credential-like file: {rel}")
        return candidate

    # ---------------------------------------------------------------- tools
    def read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._safe(str(args.get("path", "")))
        if path.is_dir():
            return ToolResult(
                "read_file", False,
                f"{args.get('path')} is a directory - use list_files instead",
                error="is_directory",
            )
        if not path.exists():
            return ToolResult("read_file", False, f"file not found: {args.get('path')}", error="not_found")
        text = path.read_text(encoding="utf-8", errors="replace")[:_READ_LIMIT]
        lines = text.splitlines()
        note = ""
        start_raw = args.get("start_line")
        end_raw = args.get("end_line")
        has_range = start_raw is not None or end_raw is not None
        # models pass ranges in several shapes: 1138, "1138", [1138, 1200], "[1138, 1200]"
        if isinstance(start_raw, str) and re.fullmatch(r"\[\s*\d+\s*,\s*\d+\s*\]", start_raw.strip()):
            nums = [int(x) for x in re.findall(r"\d+", start_raw)]
            end_raw = nums[1] if len(nums) > 1 else end_raw
            start_raw = nums[0]
        if isinstance(start_raw, list):
            end_raw = start_raw[1] if len(start_raw) > 1 else end_raw
            start_raw = start_raw[0]
        if not has_range:
            if len(text) > 30_000:
                return ToolResult(
                    "read_file", False,
                    f"{args.get('path')} is large ({len(text)} chars). Call read_file with "
                    "start_line and end_line to read a section of about 100-200 lines.",
                    error="use_ranges",
                )
            start, end = 1, len(lines)
        else:
            try:
                start = max(1, int(start_raw) if start_raw is not None else 1)
                end = int(end_raw) if end_raw is not None else min(start + 200, len(lines))
            except (TypeError, ValueError):
                start, end = 1, min(200, len(lines))
                note = "[range could not be parsed; showing the first 200 lines]\n"
            span = end - start + 1
            if span > 250:
                end = start + 249
                note = f"[range clamped to lines {start}-{end}; read a narrower range for detail]\n"
        selected = lines[start - 1 : end]
        body = "\n".join(f"{n:>5}  {line}" for n, line in enumerate(selected, start=start))
        body, truncated = _clip(note + body)
        return ToolResult("read_file", True, body, truncated=truncated)

    def list_files(self, args: dict[str, Any]) -> ToolResult:
        rel = str(args.get("path") or ".")
        base = self._safe(rel)
        if not base.exists():
            return ToolResult("list_files", False, f"not found: {rel}", error="not_found")
        out: list[str] = []
        for path in sorted(base.rglob("*")):
            if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in path.parts):
                continue
            out.append(str(path.relative_to(self.workdir)).replace("\\", "/"))
            if len(out) >= _MAX_LIST:
                break
        body, truncated = _clip("\n".join(out))
        return ToolResult("list_files", True, body, truncated=truncated)

    def search_code(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return ToolResult("search_code", False, "pattern is required", error="bad_args")
        try:
            regex = re.compile(pattern, re.I)
        except re.error as exc:
            return ToolResult("search_code", False, f"invalid regex: {exc}", error="bad_regex")
        hits: list[str] = []
        for path in self.workdir.rglob("*.py"):
            if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(path.relative_to(self.workdir)).replace("\\", "/")
            for n, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                    if len(hits) >= 80:
                        body, truncated = _clip("\n".join(hits))
                        return ToolResult("search_code", True, body, truncated=truncated)
        body, truncated = _clip("\n".join(hits) or "no matches")
        return ToolResult("search_code", True, body, truncated=truncated)

    def str_replace_file(self, args: dict[str, Any]) -> ToolResult:
        """Exact-snippet replacement. No diff format for the model to get wrong."""
        path = self._safe(str(args.get("path", "")))
        old = args.get("old_string")
        new = args.get("new_string")
        if not path.exists():
            return ToolResult("str_replace_file", False, f"file not found: {args.get('path')}", error="not_found")
        if not isinstance(old, str) or not old:
            return ToolResult("str_replace_file", False, "old_string is required", error="bad_args")
        if not isinstance(new, str):
            return ToolResult("str_replace_file", False, "new_string is required", error="bad_args")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(
                "str_replace_file", False,
                "old_string not found in the file - read the file again and copy the text exactly",
                error="not_found",
            )
        if count > 1:
            return ToolResult(
                "str_replace_file", False,
                f"old_string occurs {count} times - include more surrounding text so it is unique",
                error="not_unique",
            )
        if old == new:
            return ToolResult("str_replace_file", False, "old_string and new_string are identical", error="no_change")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult("str_replace_file", True, f"replaced 1 occurrence in {path.name}")

    def apply_patch(self, args: dict[str, Any]) -> ToolResult:
        diff = str(args.get("diff") or "")
        if not diff.strip():
            return ToolResult("apply_patch", False, "diff is required", error="bad_args")
        if not (self.workdir / ".git").exists():
            return ToolResult("apply_patch", False, "workdir is not a git checkout", error="no_git")
        tmp = self.workdir / ".mente_patch.diff"
        tmp.write_text(diff, encoding="utf-8")
        # models emit diffs in several conventions: a/ b/ prefixes or bare paths,
        # and on Windows the worktree may be CRLF while the diff is LF. Rather
        # than reject a correct fix over format, try the tolerant combinations.
        attempts = [
            ["git", "apply", "--ignore-whitespace", "--recount", "-p1", str(tmp)],
            ["git", "apply", "--ignore-whitespace", "--recount", "-p0", str(tmp)],
        ]
        proc = None
        try:
            for cmd in attempts:
                try:
                    proc = subprocess.run(
                        cmd, cwd=str(self.workdir), capture_output=True, text=True, timeout=60,
                    )
                except subprocess.TimeoutExpired:
                    return ToolResult("apply_patch", False, "patch timed out", error="timeout")
                if proc.returncode == 0:
                    break
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        if proc is None or proc.returncode != 0:
            err = (proc.stderr if proc else "") or (proc.stdout if proc else "") or ""
            body, trunc = _clip(err[-1500:])
            return ToolResult(
                "apply_patch", False,
                f"patch rejected:\n{body}\nHint: use a unified diff with ---/+++ headers.",
                truncated=trunc, error="apply_failed",
            )
        return ToolResult("apply_patch", True, "patch applied")

    def run_tests(self, args: dict[str, Any]) -> ToolResult:
        cmd = str(args.get("command") or self.test_command)
        # only test runners: the runtime, not the model, decides what may execute
        if "pytest" not in cmd:
            return ToolResult("run_tests", False, "only pytest commands may be run here", error="not_allowed")
        # a bare "python" resolves to whatever is on PATH, which may be too old to
        # parse the repo. Always run the tests with the collector's interpreter.
        if cmd.lstrip().startswith("python"):
            cmd = f'"{sys.executable}"' + cmd.lstrip()[len("python") :]
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=str(self.workdir), capture_output=True,
                text=True, timeout=self.timeout_s,
                env={**os.environ, "PYTHONPATH": str(self.workdir), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult("run_tests", False, f"tests timed out after {self.timeout_s}s", error="timeout")
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-_MAX_TOOL_OUTPUT:]
        body, truncated = _clip(tail)
        verdict = "passed" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
        return ToolResult("run_tests", proc.returncode == 0, f"tests {verdict}\n{body}", truncated=truncated)

    # ---------------------------------------------------------------- dispatch
    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        table = {
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_code": self.search_code,
            "str_replace_file": self.str_replace_file,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
        }
        fn = table.get(name)
        if fn is None:
            return ToolResult(name, False, f"unknown tool: {name}", error="unknown_tool")
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        try:
            return fn(arguments)
        except ToolError as exc:
            return ToolResult(name, False, str(exc), error="refused")
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the run
            return ToolResult(name, False, f"{type(exc).__name__}: {exc}", error="exception")


def sys_executable() -> str:
    return "python"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_file",
            "description": (
                "Edit a file by replacing an exact unique snippet. old_string must appear "
                "EXACTLY once in the file (copy it precisely, including indentation). "
                "This is the preferred way to edit files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "old_string": {"type": "string", "description": "The exact text to replace."},
                    "new_string": {"type": "string", "description": "The replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository, with line numbers. Use start_line/end_line for big files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List every file in the repository (relative paths).",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search all Python files for a regular expression. Returns path:line:match.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to the repository. This is how you edit files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": {"type": "string", "description": "A unified diff, including ---/+++ headers."},
                },
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and return the result. Always do this before claiming success.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Optional pytest command override."}},
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a coding agent working inside a Python repository. Fix the issue the user "
    "describes. Read the relevant source before editing. Edit files with str_replace_file: "
    "copy the old text EXACTLY as it appears (with its indentation) and provide the corrected "
    "replacement. Make the smallest correct change. Run the tests with run_tests before you "
    "finish, and keep going until they pass. If you cannot fix the issue, say so plainly - "
    "never claim success you have not verified. Tools are listed in the request."
)


# --------------------------------------------------------------------------- runner
@dataclass
class AgentConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "cohere/north-mini-code:free"
    temperature: float = 0.7
    top_p: float = 0.95
    max_turns: int = 25
    request_timeout_s: int = 180
    max_retries: int = 5
    referer: str = "https://menteeai.org"
    title: str = "MenteE Code collector"


class AgentRunner:
    """Run one task through the agent loop and return a raw trajectory."""

    def __init__(self, cfg: AgentConfig, executor: ToolExecutor):
        self.cfg = cfg
        self.executor = executor

    # ------------------------------------------------------------------ http
    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": messages,
                "tools": TOOLS,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.cfg.referer,
            "X-Title": self.cfg.title,
        }
        last = ""
        for attempt in range(self.cfg.max_retries):
            req = urllib.request.Request(
                f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                data=body, headers=headers, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
                retry_after = exc.headers.get("Retry-After")
                if exc.code == 429:
                    wait = float(retry_after) if retry_after else min(60.0, (2**attempt) * 2 + random.random() * 2)
                    time.sleep(wait)
                    continue
                if exc.code >= 500:
                    time.sleep(min(30.0, (2**attempt) * 2))
                    continue
                raise RuntimeError(last) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(30.0, (2**attempt) * 2))
                continue
            # A 200 response can still carry an error object (overloaded free
            # provider, upstream timeout). Treat it as a retryable provider
            # error instead of letting a KeyError('choices') kill the run.
            if not isinstance(payload, dict) or not payload.get("choices"):
                detail = ""
                if isinstance(payload, dict):
                    err = payload.get("error") or payload
                    if isinstance(err, dict):
                        detail = str(err.get("message") or err.get("code") or err)[:300]
                    else:
                        detail = str(err)[:300]
                last = f"provider returned no choices: {detail or 'empty response'}"
                time.sleep(min(20.0, (2**attempt) * 2 + random.random() * 2))
                continue
            return payload
        raise RuntimeError(f"request failed after {self.cfg.max_retries} attempts: {last}")
    # ------------------------------------------------------------------ loop
    def run(self, task: dict[str, Any]) -> tuple[RawTrajectory, dict[str, Any]]:
        """Run the agent on one task. Returns (raw_trajectory, run_metadata)."""
        executor = ToolExecutor(
            Path(task["workdir"]),
            test_command=task.get("test_command", ""),
            timeout_s=max(120, self.cfg.request_timeout_s),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["instruction"]},
        ]
        common: list[CommonMessage] = []
        usage_in = usage_out = 0
        stop_reason = "completed"
        consecutive_errors = 0
        last_test_result: ToolResult | None = None
        last_provider_error: str | None = None
        started = time.time()

        for _turn in range(self.cfg.max_turns):
            try:
                resp = self._chat(messages)
            except RuntimeError as exc:
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    stop_reason = "error"
                    last_provider_error = str(exc)
                    common.append(CommonMessage(role="assistant", content=f"run stopped: {exc}"))
                    break
                # tell the model and let it continue with a simpler step
                messages.append(
                    {
                        "role": "user",
                        "content": f"The provider returned an error ({str(exc)[:200]}). "
                        "Continue with a simpler next step.",
                    }
                )
                continue
            consecutive_errors = 0

            choice = resp["choices"][0]
            msg = choice["message"]
            usage = resp.get("usage", {}) or {}
            usage_in += int(usage.get("prompt_tokens") or 0)
            usage_out += int(usage.get("completion_tokens") or 0)

            tool_calls = msg.get("tool_calls") or []
            # some models emit the same call twice within one turn; executing both
            # would record duplicate ids and poison the row. Normalise here.
            seen_calls: set[tuple[str, str]] = set()
            unique_calls = []
            for tc in tool_calls:
                key = (tc["function"]["name"], str(tc["function"].get("arguments", "{}")))
                if key in seen_calls:
                    continue
                seen_calls.add(key)
                unique_calls.append(tc)
            tool_calls = unique_calls
            content = msg.get("content") or ""

            if not tool_calls:
                common.append(CommonMessage(role="assistant", content=content))
                break

            assistant_calls = []
            results: list[dict[str, Any]] = []
            made: set[tuple[str, str]] = set()
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"].get("arguments", "{}"),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError as exc:
                    args = {}
                    parse_err = f"arguments were not valid JSON: {exc}"
                else:
                    parse_err = None
                if not isinstance(args, dict):
                    args = {"value": args}

                repeat_key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if parse_err:
                    result = ToolResult(name, False, parse_err, error="bad_json")
                else:
                    result = executor.execute(name, args)
                if name == "run_tests":
                    last_test_result = result
                if repeat_key in made and result.ok:
                    # tell the model the call is a repeat so it can change approach -
                    # weak models loop silently otherwise
                    result = ToolResult(
                        result.name, result.ok,
                        result.content + "\n\n[note: you already made this exact call. "
                        "Do something different: read a different range, search, or apply a patch.]",
                        truncated=result.truncated,
                    )
                made.add(repeat_key)
                common.append(
                    CommonMessage(
                        role="assistant",
                        content=content,
                        tool_calls=[
                            {
                                "id": tc["id"],
                                "name": name,
                                "arguments": args,
                                "arguments_raw": raw_args if isinstance(raw_args, str) else json.dumps(args),
                            }
                        ],
                    )
                )
                common.append(
                    CommonMessage(
                        role="tool_result",
                        content=result.content,
                        tool_results=[
                            {
                                "tool_call_id": tc["id"],
                                "name": name,
                                "content": result.content,
                                "truncated": result.truncated,
                            }
                        ],
                    )
                )
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result.content,
                    }
                )
                # reset so a second call in the same turn does not duplicate prose
                content = ""
            messages.extend(results)
        else:
            stop_reason = "max_turns"

        meta = {
            "duration_s": round(time.time() - started, 2),
            "total_tokens": {"input": usage_in, "output": usage_out},
            "stop_reason": stop_reason,
            "verified_by_agent": last_test_result.ok if last_test_result else None,
            "provider_error": last_provider_error,
        }
        # outcome evidence: the agent's own last test run. Honest and cheap, and
        # the honest_conclusion check cross-examines it against what the model claims.
        verification: dict[str, Any] = {"status": "skipped", "method": "not_run"}
        if last_test_result is not None:
            verification = {
                "status": "pass" if last_test_result.ok else "fail",
                "method": "agent_run_tests",
                "verifier": "runtime",
                "detail": last_test_result.content[:400],
            }

        raw = RawTrajectory(
            task=task,
            messages=common,
            tool_schemas=TOOLS,
            harness="mente-agent",
            model_id=self.cfg.model,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            verification=verification,
            generator={
                "source_id": "mente_synth_v1",
                "source_url": self.cfg.base_url,
                "license": "see teacher terms",
                "license_verified": True,
                "trainable": True,
                "collected_by": "",
                "prompt_template": "mente_agent_v1",
                "teacher": {
                    "model": self.cfg.model,
                    "revision": "unknown",
                    "license": "provider terms",
                    "sampling": {"temperature": self.cfg.temperature, "top_p": self.cfg.top_p},
                },
            },
            advertised_tools=[t["function"]["name"] for t in TOOLS],
        )
        return raw, meta
