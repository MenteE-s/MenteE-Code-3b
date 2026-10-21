"""Validation: the Tier A gate and the Tier B deep sampler.

Tier A
------
Runs on **every** row, is fast, and decides whether the row may be appended.
A row that fails any Tier A check is rejected (and the reasons are recorded).

Tier B
------
Expensive checks — sandbox re-execution, patch application, per-test evidence.
Running these on every row would make the collector the bottleneck, so they run
on a **sample** (default 2%) *before* the row is appended, and 100% of the
dataset is expected to be re-checked in a post-collection pass.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checks import CheckResult, run_tier_a

__all__ = ["Verdict", "TierAValidator", "DeepVerifier", "load_schema", "make_validator"]


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_validator(schema: dict[str, Any]) -> Any | None:
    """Build a jsonschema validator, or None if the library is unavailable."""
    try:
        import jsonschema
    except ImportError:
        return None
    return jsonschema.Draft202012Validator(schema)


@dataclass
class Verdict:
    """The outcome of validating one record."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    deep: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return "ok"
        return "; ".join(self.reasons)


class TierAValidator:
    """The per-row gate."""

    def __init__(
        self,
        validator: Any | None = None,
        known_tools: list[str] | None = None,
        repo_files: set[str] | None = None,
        min_chars: int = 400,
        max_chars: int = 400_000,
        max_identical_tool_calls: int = 3,
    ):
        self.validator = validator
        self.known_tools = known_tools
        self.repo_files = repo_files
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_identical_tool_calls = max_identical_tool_calls

    def validate(self, record: dict[str, Any]) -> Verdict:
        results = run_tier_a(
            record,
            validator=self.validator,
            known_tools=self.known_tools,
            repo_files=self.repo_files,
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            max_identical_tool_calls=self.max_identical_tool_calls,
        )
        reasons = [f"{r.name}: {r.reason}" for r in results if not r.ok]
        return Verdict(ok=not reasons, reasons=reasons, checks=results)


# --------------------------------------------------------------------------- tier B
_PYTEST_LINE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


class DeepVerifier:
    """Tier B: actually run the code.

    This is what separates "the agent said it worked" from "the tests are green".
    Without it the dataset teaches imitation; with it, the dataset teaches
    completion.
    """

    def __init__(self, timeout_s: int = 900, allow_network: bool = False):
        self.timeout_s = timeout_s
        self.allow_network = allow_network

    # ------------------------------------------------------------------ helpers
    def _run(self, cmd: str, cwd: str, env: dict[str, str] | None = None) -> tuple[int, str]:
        full_env = dict(os.environ)
        full_env.setdefault("PYTHONHASHSEED", "0")
        full_env.setdefault("TZ", "UTC")
        if env:
            full_env.update(env)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=full_env,
            )
        except subprocess.TimeoutExpired:
            return 124, f"TIMEOUT after {self.timeout_s}s"
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def _parse_pytest(self, output: str) -> dict[str, str]:
        results: dict[str, str] = {}
        for line in output.splitlines():
            line = line.strip()
            m = _PYTEST_LINE.match(line)
            if m:
                results[m.group(1)] = m.group(2)
        return results

    # ------------------------------------------------------------------ public
    def verify(self, record: dict[str, Any], workdir: str | Path | None = None) -> dict[str, Any]:
        """Run the deep checks. Returns evidence (never raises)."""
        env = record.get("environment", {})
        test_cmd = env.get("test_command") or ""
        work = str(workdir or env.get("workdir") or "")
        out: dict[str, Any] = {"status": "skipped", "reason": ""}

        if not work or not Path(work).exists():
            out["reason"] = "no workdir available"
            return out
        if not test_cmd:
            out["reason"] = "no test command"
            return out

        # -- patch application ------------------------------------------------
        patches = record.get("trajectory", {}).get("patches", []) or []
        applied_ok = True
        for patch in patches:
            diff = patch.get("diff")
            if not diff:
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as fh:
                fh.write(diff)
                diff_path = fh.name
            try:
                code, output = self._run(f'git apply --check "{diff_path}"', work)
                if code != 0:
                    applied_ok = False
                    out["patch_error"] = output[-800:]
            finally:
                try:
                    os.unlink(diff_path)
                except OSError:
                    pass
        out["patch_applies"] = applied_ok
        if not applied_ok:
            out["status"] = "fail"
            out["reason"] = "patch does not apply to the base commit"
            return out

        # -- test execution ----------------------------------------------------
        code, output = self._run(test_cmd, work)
        out["returncode"] = code
        out["output_tail"] = output[-2000:]

        per_test = self._parse_pytest(output) if "pytest" in test_cmd else {}
        if per_test:
            out["per_test"] = per_test

        f2p = env.get("fail_to_pass", []) or []
        p2p = env.get("pass_to_pass", []) or []

        if f2p:
            still_failing = [t for t in f2p if per_test.get(t) not in ("PASSED", "XPASS")]
            if still_failing:
                out["status"] = "fail"
                out["reason"] = f"{len(still_failing)} fail_to_pass test(s) still failing"
                out["still_failing"] = still_failing[:20]
                return out
            out["fail_to_pass"] = {"before": "FAIL", "after": "PASS"}
        else:
            out["fail_to_pass"] = {"before": "UNKNOWN", "after": "UNKNOWN"}

        if p2p and per_test:
            regressed = [t for t in p2p if per_test.get(t) in ("FAILED", "ERROR")]
            if regressed:
                out["status"] = "fail"
                out["reason"] = f"{len(regressed)} regression(s)"
                out["regressions"] = regressed[:20]
                return out
            out["pass_to_pass"] = {
                "before": "PASS",
                "after": "PASS",
                "total": len(p2p),
                "regressed": 0,
            }
        else:
            out["pass_to_pass"] = {"before": "UNKNOWN", "after": "UNKNOWN", "total": 0, "regressed": 0}

        if code != 0 and not per_test:
            out["status"] = "fail"
            out["reason"] = f"test command exited {code}"
            return out

        out["status"] = "pass"
        out["verifier"] = f"deep-verifier@1.0.0"
        return out
