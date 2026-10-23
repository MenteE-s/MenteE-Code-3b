"""Trajectory sources.

A *source* yields :class:`~src.collect.normalizer.RawTrajectory` objects. Two
implementations ship here:

``HarnessSource``
    Drives a live coding agent (currently OpenCode's headless server) against a
    task list and captures the session.

``ReplaySource``
    Reads trajectories that were produced elsewhere — an exported OpenCode session,
    a public dataset, a beta opt-in capture — and hands them to the same
    validator and appender.

Both produce the same intermediate shape, so the validator, the appender, the
integrity checks and the reporting are completely unaware of which path produced
the data. That is what makes the collector reusable across the weekend, the
dataset import and the beta.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from .normalizer import CommonMessage, GenericAdapter, OpenCodeAdapter, RawTrajectory

__all__ = ["Task", "TrajectorySource", "HarnessSource", "ReplaySource", "load_tasks"]


@dataclass
class Task:
    """One unit of work for the agent to attempt."""

    task_id: str
    instruction: str
    repo: str = ""
    base_commit: str = ""
    setup_script: str = ""
    test_command: str = ""
    language: str = "unknown"
    task_type: str = "bug_fix"
    difficulty: str = "unlabelled"
    workdir: str = ""
    fail_to_pass: list[str] | None = None
    pass_to_pass: list[str] | None = None
    issue_url: str | None = None
    extra: dict[str, Any] | None = None

    def to_record_task(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "setup_script": self.setup_script,
            "test_command": self.test_command,
            "language": self.language,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
        }
        if self.fail_to_pass:
            out["fail_to_pass"] = self.fail_to_pass
        if self.pass_to_pass:
            out["pass_to_pass"] = self.pass_to_pass
        if self.issue_url:
            out["issue_url"] = self.issue_url
        if self.extra:
            out.update(self.extra)
        return out


def load_tasks(path: str | Path) -> list[Task]:
    """Load a task list from JSONL.

    Each line needs at least ``task_id`` and ``instruction``; everything else is
    optional and defaults sensibly.
    """
    tasks: list[Task] = []
    p = Path(path)
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{p}:{i} is not valid JSON: {exc}") from exc
        tasks.append(
            Task(
                task_id=str(obj.get("task_id") or obj.get("id") or f"task_{i:06d}"),
                instruction=obj.get("instruction") or obj.get("prompt") or "",
                repo=obj.get("repo", ""),
                base_commit=obj.get("base_commit", ""),
                setup_script=obj.get("setup_script", ""),
                test_command=obj.get("test_command", ""),
                language=obj.get("language", "unknown"),
                task_type=obj.get("task_type", "bug_fix"),
                difficulty=obj.get("difficulty", "unlabelled"),
                workdir=obj.get("workdir", ""),
                fail_to_pass=obj.get("fail_to_pass"),
                pass_to_pass=obj.get("pass_to_pass"),
                issue_url=obj.get("issue_url"),
                extra=obj.get("extra"),
            )
        )
    return tasks


class TrajectorySource(Protocol):
    """Anything that can yield raw trajectories."""

    def __iter__(self) -> Iterator[RawTrajectory]: ...


# --------------------------------------------------------------------------- live
class HarnessSource:
    """Drive OpenCode's headless server over one task list.

    The server is the documented headless HTTP interface::

        opencode serve --port 4096 --hostname 127.0.0.1

    and the flow per task is: create a session, prompt it, then read the
    structured transcript back.
    """

    def __init__(
        self,
        tasks: list[Task],
        server_url: str = "http://127.0.0.1:4096",
        auth_user: str = "opencode",
        auth_password: str = "",
        model: str = "",
        agent: str = "",
        timeout_s: int = 1800,
        tool_schemas: list[dict[str, Any]] | None = None,
        raw_args_by_call: dict[str, str] | None = None,
        generator: dict[str, Any] | None = None,
    ):
        self.tasks = tasks
        self.server_url = server_url.rstrip("/")
        self.auth_user = auth_user
        self.auth_password = auth_password
        self.model = model
        self.agent = agent
        self.timeout_s = timeout_s
        self.tool_schemas = tool_schemas or []
        self.raw_args_by_call = raw_args_by_call or {}
        self.generator = generator or {}

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        url = f"{self.server_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.auth_password:
            token = base64.b64encode(
                f"{self.auth_user}:{self.auth_password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        if not body.strip():
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    # ------------------------------------------------------------------ source
    def __iter__(self) -> Iterator[RawTrajectory]:
        for task in self.tasks:
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0 = time.time()
            try:
                session = self._request("POST", "/session", {})
                sid = session.get("id") if isinstance(session, dict) else None
                if not sid:
                    continue
                self._request(
                    "POST",
                    f"/session/{sid}/message",
                    {"parts": [{"type": "text", "text": task.instruction}]},
                )
                transcript = self._request("GET", f"/session/{sid}/message") or []
            except (urllib.error.URLError, OSError, TimeoutError):
                continue

            messages = OpenCodeAdapter.to_common(
                transcript if isinstance(transcript, list) else [],
                self.raw_args_by_call,
            )
            if not messages:
                continue

            record_task = task.to_record_task()
            record_task["duration_s"] = round(time.time() - t0, 2)
            generator = dict(self.generator)
            generator.setdefault("harness", "opencode")
            if self.agent:
                generator["agent_variant"] = self.agent
            yield RawTrajectory(
                task=record_task,
                messages=messages,
                tool_schemas=self.tool_schemas,
                harness="opencode",
                model_id=self.model,
                started_at=started,
                ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                generator=generator,
                advertised_tools=[t.get("name", "") for t in self.tool_schemas],
            )


# --------------------------------------------------------------------------- replay
class ReplaySource:
    """Yield trajectories from records that were produced elsewhere.

    Accepts JSONL where each line is either:

    * an OpenCode server response (``[{info, parts}, ...]``) plus a ``task`` field, or
    * a plain message list plus a ``task`` field, or
    * a record that already has a ``messages`` key in the common shape.
    """

    def __init__(
        self,
        path: str | Path,
        generator: dict[str, Any] | None = None,
        format: str = "auto",
    ):
        self.path = Path(path)
        self.generator = generator or {}
        self.format = format

    def __iter__(self) -> Iterator[RawTrajectory]:
        p = self.path
        if p.is_dir():
            files = sorted(p.glob("*.json"))
        else:
            files = [p]
        for file in files:
            text = file.read_text(encoding="utf-8")
            items: list[Any]
            if file.suffix == ".jsonl":
                items = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                loaded = json.loads(text)
                items = loaded if isinstance(loaded, list) else [loaded]
            for item in items:
                raw = self._to_raw(item, file)
                if raw is not None:
                    yield raw

    def _to_raw(self, item: Any, file: Path) -> RawTrajectory | None:
        if not isinstance(item, dict):
            return None
        task = item.get("task") or item.get("task_spec") or {}
        generator = dict(self.generator)
        generator.setdefault("source_id", file.stem)

        if "messages" in item and isinstance(item["messages"], list):
            messages = GenericAdapter.to_common(item["messages"], item.get("raw_args_by_call"))
        elif "info" in item or "parts" in item:
            messages = OpenCodeAdapter.to_common([item], item.get("raw_args_by_call"))
        elif isinstance(item.get("transcript"), list):
            messages = OpenCodeAdapter.to_common(item["transcript"], item.get("raw_args_by_call"))
        else:
            return None

        if not messages:
            return None

        return RawTrajectory(
            task=task,
            messages=messages,
            tool_schemas=item.get("tool_schemas", []) or [],
            harness=generator.get("harness", "replay"),
            model_id=item.get("model", ""),
            started_at=item.get("started_at", ""),
            ended_at=item.get("ended_at", ""),
            verification=item.get("verification", {}) or {},
            generator=generator,
            advertised_tools=[t.get("name", "") for t in (item.get("tool_schemas") or [])],
        )
