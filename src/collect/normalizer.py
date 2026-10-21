"""Normalisation: harness-native transcripts -> canonical schema v1.

The pipeline is deliberately split in two:

1. **Adapter** — turns a harness-specific transcript into a common message shape.
   One adapter per harness. The rest of the pipeline never sees harness specifics.
2. **Builder** — turns the common message shape into a schema v1 record.

This is what makes the collector reusable: adding a harness means writing one
adapter, not touching the validator, the appender or the checks.

Common message shape
--------------------
``{"role", "content", "tool_calls", "tool_results"}`` where ``tool_calls`` entries
are ``{"id", "name", "arguments", "arguments_raw"}`` and ``tool_results`` entries
are ``{"tool_call_id", "name", "content", "truncated"}``.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "RawTrajectory",
    "CommonMessage",
    "OpenCodeAdapter",
    "GenericAdapter",
    "build_record",
    "find_tool_call_blocks",
    "merge_tool_results",
]


# --------------------------------------------------------------------------- model
@dataclass
class CommonMessage:
    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RawTrajectory:
    """What a source produces, before normalisation."""

    task: dict[str, Any]
    messages: list[CommonMessage] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    harness: str = "generic"
    model_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    repo_files: set[str] | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    generator: dict[str, Any] = field(default_factory=dict)
    advertised_tools: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- utils
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def find_tool_call_blocks(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from a raw model response.

    Parses by brace balance rather than a regex, because the arguments object
    contains nested braces and a non-greedy regex would truncate it.

    Returns a list of ``{"name": str, "arguments": dict, "arguments_raw": str}``.
    """
    out: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, pos)
        if start == -1:
            break
        body_start = start + len(_TOOL_CALL_OPEN)
        close = text.find(_TOOL_CALL_CLOSE, body_start)
        if close == -1:
            break
        body = text[body_start:close].strip()
        parsed = _balanced_json(body)
        if parsed is not None and isinstance(parsed, dict):
            name = parsed.get("name")
            args = parsed.get("arguments", {})
            if not isinstance(args, dict):
                args = {"value": args}
            out.append(
                {
                    "name": name,
                    "arguments": args,
                    "arguments_raw": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                }
            )
        pos = close + len(_TOOL_CALL_CLOSE)
    return out


def _balanced_json(body: str) -> Any | None:
    """Parse the first complete JSON object in ``body``."""
    body = body.lstrip()
    if not body.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[: i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


def _canonical_args(args: Any) -> str:
    if not isinstance(args, dict):
        args = {"value": args}
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


def merge_tool_results(messages: list[CommonMessage]) -> list[CommonMessage]:
    """Fold tool results into the assistant turn that issued the call.

    A harness may report a tool result as a separate message; the canonical schema
    wants the call and its result adjacent so the pair survives tokenisation.
    """
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for msg in messages:
        for res in msg.tool_results:
            tid = res.get("tool_call_id")
            if tid:
                by_id[tid] = res
                order.append(tid)

    out: list[CommonMessage] = []
    for msg in messages:
        if msg.role == "tool_result" or (msg.tool_results and not msg.tool_calls):
            continue
        if msg.tool_calls:
            for call in msg.tool_calls:
                res = by_id.get(call.get("id", ""))
                if res is not None:
                    msg.tool_results.append(res)
        out.append(msg)
    return out


# --------------------------------------------------------------------------- adapters
class OpenCodeAdapter:
    """Adapter for OpenCode's server API.

    ``GET /session/:id/message`` returns ``{info: Message, parts: Part[]}[]``.
    Tool parts carry ``state`` with ``status`` (``running`` / ``completed`` /
    ``error``), ``input`` (the arguments) and ``output``.
    """

    name = "opencode"

    @classmethod
    def to_common(
        cls,
        messages_response: list[dict[str, Any]],
        raw_args_by_call: dict[str, str] | None = None,
    ) -> list[CommonMessage]:
        raw_args_by_call = raw_args_by_call or {}
        out: list[CommonMessage] = []

        for entry in messages_response:
            info = entry.get("info", {}) or {}
            parts = entry.get("parts", []) or []
            role = info.get("role", "assistant")

            if role == "user":
                text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
                out.append(CommonMessage(role="user", content=text))
                continue

            text_parts: list[str] = []
            calls: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []

            for part in parts:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                elif ptype == "reasoning":
                    # The chosen teacher is non-thinking, so reasoning should be
                    # absent. If it appears we keep the fact, we do not drop it.
                    text_parts.append(part.get("text", ""))
                elif ptype == "tool":
                    state = part.get("state", {}) or {}
                    status = state.get("status", "completed")
                    cid = part.get("callID") or part.get("id") or ""
                    name = part.get("tool") or part.get("name") or ""
                    args = state.get("input", {}) or {}
                    calls.append(
                        {
                            "id": cid,
                            "name": name,
                            "arguments": args if isinstance(args, dict) else {"value": args},
                            "arguments_raw": raw_args_by_call.get(cid, _canonical_args(args)),
                        }
                    )
                    if status == "completed":
                        results.append(
                            {
                                "tool_call_id": cid,
                                "name": name,
                                "content": _stringify(state.get("output")),
                                "truncated": _is_truncated(state),
                            }
                        )
                    elif status == "error":
                        results.append(
                            {
                                "tool_call_id": cid,
                                "name": name,
                                "content": _stringify(state.get("error") or "tool error"),
                                "truncated": False,
                            }
                        )
                    # status == "running" means the run was cut mid-call; the
                    # builder will record stop_reason accordingly.

            content = "\n".join(t for t in text_parts if t).strip()
            if calls:
                out.append(CommonMessage(role="assistant", content=content, tool_calls=calls))
                for res in results:
                    out.append(
                        CommonMessage(
                            role="tool_result",
                            content=res["content"],
                            tool_results=[res],
                        )
                    )
            elif content:
                out.append(CommonMessage(role="assistant", content=content))

        return merge_tool_results(out)


class GenericAdapter:
    """Adapter for a plain message list.

    Expected shape per message::

        {"role": "assistant",
         "content": "...",
         "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}],
         "tool_results": [{"tool_call_id": "...", "name": "...", "content": "...",
                           "truncated": false}]}
    """

    name = "generic"

    @classmethod
    def to_common(
        cls,
        messages: list[dict[str, Any]],
        raw_args_by_call: dict[str, str] | None = None,
    ) -> list[CommonMessage]:
        raw_args_by_call = raw_args_by_call or {}
        out: list[CommonMessage] = []
        for m in messages:
            calls = []
            for call in m.get("tool_calls", []) or []:
                args = call.get("arguments", {}) or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                cid = call.get("id", "")
                calls.append(
                    {
                        "id": cid,
                        "name": call.get("name", ""),
                        "arguments": args,
                        "arguments_raw": call.get("arguments_raw")
                        or raw_args_by_call.get(cid)
                        or _canonical_args(args),
                    }
                )
            results = []
            for res in m.get("tool_results", []) or []:
                results.append(
                    {
                        "tool_call_id": res.get("tool_call_id", ""),
                        "name": res.get("name", ""),
                        "content": _stringify(res.get("content")),
                        "truncated": bool(res.get("truncated", False)),
                    }
                )
            out.append(
                CommonMessage(
                    role=m.get("role", "assistant"),
                    content=m.get("content", "") or "",
                    tool_calls=calls,
                    tool_results=results,
                )
            )
        return merge_tool_results(out)


# --------------------------------------------------------------------------- helpers
def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _is_truncated(state: dict[str, Any]) -> bool:
    """Detect a tool output that the harness shortened.

    OpenCode truncates long tool output before storing it. A truncated result means
    the model never saw the rest of the file, so training on it teaches the model
    to hallucinate the remainder. The Tier A check rejects these rows.
    """
    meta = state.get("metadata") or {}
    for key in ("truncated", "is_truncated", "output_truncated"):
        if meta.get(key) is True:
            return True
    if state.get("truncated") is True:
        return True
    output = state.get("output")
    if isinstance(output, str) and output.endswith("\n... (truncated)"):
        return True
    return False


# --------------------------------------------------------------------------- builder
def build_record(
    raw: RawTrajectory,
    record_id: str,
    now: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build a schema v1 record from a normalised trajectory.

    Returns ``(record, problems)``. ``record`` is ``None`` when the trajectory
    could not be turned into a record at all.
    """
    problems: list[str] = []
    messages = raw.messages
    if not messages:
        return None, ["no messages"]

    # -- turns ---------------------------------------------------------------
    turns: list[dict[str, Any]] = []
    index = 0
    stop_reason = "completed"

    # Pre-pass: collect every tool result by call id. A harness may report the
    # result *after* the assistant turn that issued the call, so the results have
    # to be gathered before the turns are laid out.
    pending_results: dict[str, dict[str, Any]] = {}
    for msg in messages:
        for res in msg.tool_results:
            tid = res.get("tool_call_id")
            if tid:
                pending_results[tid] = res

    for msg in messages:
        if msg.role == "tool_result":
            continue
        if msg.role == "assistant" and msg.tool_calls:
            # attach the result that belongs to each call
            for call in msg.tool_calls:
                res = pending_results.get(call.get("id", ""))
                if res is not None and not any(
                    r.get("tool_call_id") == call.get("id") for r in msg.tool_results
                ):
                    msg.tool_results.append(res)
            turns.append(
                {
                    "index": index,
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "arguments": c["arguments"],
                            "arguments_raw": c["arguments_raw"],
                        }
                        for c in msg.tool_calls
                    ],
                }
            )
            index += 1
            for res in msg.tool_results:
                turns.append(
                    {
                        "index": index,
                        "role": "tool_result",
                        "content": res.get("content", ""),
                        "tool_call_id": res.get("tool_call_id", ""),
                        "name": res.get("name", ""),
                        "truncated": bool(res.get("truncated", False)),
                        "content_bytes": len((res.get("content") or "").encode("utf-8")),
                    }
                )
                index += 1
        else:
            turns.append(
                {
                    "index": index,
                    "role": msg.role,
                    "content": msg.content,
                }
            )
            index += 1

    if pending_results:
        stop_reason = "cancelled"

    # -- provenance ----------------------------------------------------------
    task = raw.task or {}
    teacher = raw.generator.get("teacher", {}) if raw.generator else {}
    prov: dict[str, Any] = {
        "source_id": raw.generator.get("source_id", "unknown"),
        "source_url": raw.generator.get("source_url", ""),
        "source_sha256": raw.generator.get("source_sha256", ""),
        "license": raw.generator.get("license", "unknown"),
        "license_verified": bool(raw.generator.get("license_verified", False)),
        "redistributable": bool(raw.generator.get("redistributable", False)),
        "trainable": bool(raw.generator.get("trainable", False)),
        "teacher": {
            "model": teacher.get("model", raw.model_id or "unknown"),
            "revision": teacher.get("revision", "unknown"),
            "license": teacher.get("license", raw.generator.get("license", "unknown")),
            "sampling": teacher.get("sampling", {}),
        },
        "generator": {
            "code_version": raw.generator.get("code_version", ""),
            "prompt_template": raw.generator.get("prompt_template", ""),
            "harness": raw.harness,
        },
        "collected_at": raw.started_at or (now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "collected_by": raw.generator.get("collected_by", "collector"),
    }
    if raw.generator.get("tool_name_mapping"):
        prov["generator"]["tool_name_mapping"] = raw.generator["tool_name_mapping"]
    if raw.generator.get("arguments_raw_source"):
        prov["generator"]["arguments_raw_source"] = raw.generator["arguments_raw_source"]

    # -- environment ---------------------------------------------------------
    env: dict[str, Any] = {
        "repo": task.get("repo", ""),
        "repo_license": task.get("repo_license"),
        "base_commit": task.get("base_commit", ""),
        "setup_script": task.get("setup_script", ""),
        "test_command": task.get("test_command", ""),
        "fail_to_pass": task.get("fail_to_pass", []) or [],
        "pass_to_pass": task.get("pass_to_pass", []) or [],
    }
    if task.get("repo_url"):
        env["repo_url"] = task["repo_url"]
    if task.get("runtime"):
        env["runtime"] = task["runtime"]

    # -- verification --------------------------------------------------------
    ver = dict(raw.verification or {})
    ver.setdefault("status", "skipped")
    ver.setdefault("verified_at", now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if ver.get("status") == "pass" and not ver.get("tests"):
        problems.append("verification status is pass but no test evidence was recorded")

    # -- quality -------------------------------------------------------------
    quality = {
        "verified": ver.get("status") == "pass",
        "score": 0.0,
        "checks": {
            "schema_valid": True,
            "no_truncated_tool_output": not any(t.get("truncated") for t in turns),
            "has_failure_path": _has_failure_path(turns),
            "no_secrets": True,
            "no_hallucinated_paths": True,
            "patch_applies": None,
            "regressions": len(ver.get("regressions", []) or []),
            "token_length_ok": True,
            "dedup_cluster": None,
        },
        "rejected_by": [],
        "filter_notes": [],
    }

    # -- record --------------------------------------------------------------
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "id": record_id,
        "provenance": prov,
        "task": {
            "instruction": task.get("instruction", ""),
            "language": task.get("language", "unknown"),
            "task_type": task.get("task_type", "bug_fix"),
            "difficulty": task.get("difficulty", "unlabelled"),
            "constraints": task.get("constraints", []) or [],
            "acceptance_criteria": task.get("acceptance_criteria", []) or [],
            "outcome": task.get("outcome", "unlabelled"),
        },
        "environment": env,
        "trajectory": {
            "turns": turns,
            "patches": task.get("patches", []) or [],
            "n_turns": len(turns),
            "stop_reason": task.get("stop_reason", stop_reason),
            "duration_s": task.get("duration_s"),
            "total_tokens": task.get("total_tokens", {"input": 0, "output": 0}),
        },
        "verification": ver,
        "quality": quality,
        "split": {
            "partition": "train",
            "reason": "unassigned_at_collection",
            "split_group": task.get("split_group", f"{env.get('repo','')}@{env.get('base_commit','')}"),
        },
    }
    if task.get("issue_url"):
        record["task"]["issue_url"] = task["issue_url"]

    return record, problems


def _has_failure_path(turns: list[dict[str, Any]]) -> bool:
    """True when the trajectory contains a failed step followed by a recovery.

    This is the M4 signal: a trajectory that only shows the successful final patch
    teaches SFT. One that shows the failure and the recovery teaches an agent.
    """
    seen_failure = False
    for turn in turns:
        if turn.get("role") != "tool_result":
            continue
        content = (turn.get("content") or "").lower()
        failed = any(
            marker in content
            for marker in ("error", "traceback", "failed", "exception", "not found", "no such file")
        )
        if failed:
            seen_failure = True
        elif seen_failure:
            # a subsequent successful tool result after a failure == recovery
            if any(marker in content for marker in ("passed", "ok", "success", "0 failed")):
                return True
    return False
