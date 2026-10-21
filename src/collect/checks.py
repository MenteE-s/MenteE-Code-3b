"""Tier A validation checks — the per-row gate.

Every accepted row must pass all of these. They are deliberately **fast** (< 5 ms
each) so they can run on every single record as it is generated, rather than only
on a sample. The expensive checks (sandbox re-execution, patch apply) live in
``src.collect.validate`` as Tier B.

Each check returns a :class:`CheckResult`. A row is rejected if any check fails;
all failing reasons are recorded, not just the first, because the rejection
histogram is how you debug the filters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "CheckResult",
    "SECRET_PATTERNS",
    "check_schema",
    "check_turn_sequence",
    "check_tool_call_pairing",
    "check_arguments_raw_agreement",
    "check_tool_names",
    "check_no_truncated_output",
    "check_no_secrets",
    "check_no_tool_loop",
    "check_length_bounds",
    "check_provenance",
    "check_verification_consistency",
    "check_no_hallucinated_paths",
    "check_honest_conclusion",
    "run_tier_a",
]

# --------------------------------------------------------------------------- secrets
# Deliberately conservative. A false positive costs one rejected row; a false
# negative publishes a credential in a public model.
SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"sk-[A-Za-z0-9_-]{20,}", "openai_style_key"),
    (r"sk-ant-[A-Za-z0-9_-]{20,}", "anthropic_key"),
    (r"gh[pousr]_[A-Za-z0-9]{30,}", "github_token"),
    (r"github_pat_[A-Za-z0-9_]{20,}", "github_pat"),
    (r"AKIA[0-9A-Z]{16}", "aws_access_key_id"),
    (r"ASIA[0-9A-Z]{16}", "aws_session_key"),
    (r"AIza[0-9A-Za-z_-]{30,}", "google_api_key"),
    (r"xox[baprs]-[0-9A-Za-z-]{10,}", "slack_token"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", "private_key"),
    (r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "jwt"),
    (r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][^\"'\s]{16,}[\"']", "assigned_secret"),
]

_SECRET_RES = [(re.compile(p), name) for p, name in SECRET_PATTERNS]

# Values that look like secrets but are obviously placeholders.
_PLACEHOLDERS = re.compile(
    r"(?i)(your[_-]?key|example|dummy|placeholder|redacted|xxx+|\.\.\.|<[^>]+>|changeme|placeholder)"
)

# Phrases that assert the task was completed. Deliberately narrow: a bare
# "resolved" matches "not resolved", which is the opposite of a success claim.
_SUCCESS_CLAIM = re.compile(
    r"(?i)(\ball tests (?:pass|passed)\b"
    r"|\btests? (?:now )?pass(?:ing|ed)\b"
    r"|\bi(?:'ve| have) (?:fixed|resolved|implemented|completed)\b"
    r"|\bthe (?:bug|issue|problem|failure) is (?:fixed|resolved)\b"
    r"|\bsuccessfully (?:fixed|resolved|implemented|applied)\b"
    r"|\bthis (?:should|will) (?:fix|work|resolve)\b"
    r"|\bverified\b.*\bpass(?:ing|ed)\b"
    r"|^done\.?$)"
)

# If any of these appear, the run did NOT claim success, whatever else it says.
# These are the phrases that make a "failed_but_honest" example valuable.
_NO_SUCCESS_CLAIM = re.compile(
    r"(?i)(\bnot (?:resolved|fixed|resolved yet)\b"
    r"|\bunresolved\b"
    r"|\bcould not\b"
    r"|\bcouldn't\b"
    r"|\bunable to\b"
    r"|\bdid not\b"
    r"|\bdidn't\b"
    r"|\bstill (?:fails|failing|broken|red)\b"
    r"|\btest (?:still )?fails\b"
    r"|\bwithout (?:success )?proof\b"
    r"|\bnot verified\b)"
)


@dataclass
class CheckResult:
    """Outcome of a single check."""

    name: str
    ok: bool
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _fail(name: str, reason: str, **detail: Any) -> CheckResult:
    return CheckResult(name=name, ok=False, reason=reason, detail=detail)


def _ok(name: str, **detail: Any) -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail)


# --------------------------------------------------------------------------- checks
def check_schema(record: dict, validator: Any | None = None) -> CheckResult:
    """Validate against the canonical JSON Schema."""
    if validator is None:
        return _ok("schema", note="no validator configured")
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
    if not errors:
        return _ok("schema")
    first = errors[0]
    loc = "/".join(str(x) for x in first.path) or "<root>"
    return _fail("schema", f"{loc}: {first.message}", error_count=len(errors))


def check_turn_sequence(record: dict) -> CheckResult:
    """Turn indexes must be contiguous and roles must be legal."""
    turns = record.get("trajectory", {}).get("turns", [])
    if not turns:
        return _fail("turn_sequence", "trajectory has no turns")
    for i, turn in enumerate(turns):
        if turn.get("index") != i:
            return _fail("turn_sequence", f"turn {i} has index {turn.get('index')!r}")
    roles = [t.get("role") for t in turns]
    legal = {"system", "user", "assistant", "tool_result"}
    for i, role in enumerate(roles):
        if role not in legal:
            return _fail("turn_sequence", f"turn {i} has illegal role {role!r}")
    n = len(turns)
    if record.get("trajectory", {}).get("n_turns") != n:
        return _fail(
            "turn_sequence",
            f"n_turns={record.get('trajectory', {}).get('n_turns')} but {n} turns present",
        )
    return _ok("turn_sequence", turns=n)


def check_tool_call_pairing(record: dict) -> CheckResult:
    """Every tool call must have exactly one result, and vice versa."""
    turns = record.get("trajectory", {}).get("turns", [])
    calls: dict[str, int] = {}
    results: dict[str, int] = {}
    for i, turn in enumerate(turns):
        for call in turn.get("tool_calls", []) or []:
            cid = call.get("id")
            if not cid:
                return _fail("tool_call_pairing", f"turn {i} has a tool call with no id")
            if cid in calls:
                return _fail("tool_call_pairing", f"duplicate tool call id {cid!r}")
            calls[cid] = i
        if turn.get("role") == "tool_result":
            cid = turn.get("tool_call_id")
            if not cid:
                return _fail("tool_call_pairing", f"turn {i} is a tool_result with no tool_call_id")
            if cid in results:
                return _fail("tool_call_pairing", f"duplicate tool result for {cid!r}")
            results[cid] = i
    missing = set(calls) - set(results)
    if missing:
        return _fail("tool_call_pairing", f"{len(missing)} tool call(s) have no result", ids=sorted(missing)[:10])
    orphan = set(results) - set(calls)
    if orphan:
        return _fail("tool_call_pairing", f"{len(orphan)} tool result(s) have no call", ids=sorted(orphan)[:10])
    return _ok("tool_call_pairing", calls=len(calls))


def check_arguments_raw_agreement(record: dict) -> CheckResult:
    """``arguments_raw`` must parse to exactly the parsed ``arguments``.

    This catches the re-serialisation bug: if a dict was parsed and re-serialised
    (reordering keys, changing number formatting), the training token sequence no
    longer matches what the model actually emitted.
    """
    turns = record.get("trajectory", {}).get("turns", [])
    for i, turn in enumerate(turns):
        for call in turn.get("tool_calls", []) or []:
            raw = call.get("arguments_raw")
            if raw is None:
                return _fail(
                    "arguments_raw",
                    f"turn {i} tool call {call.get('id')!r} has no arguments_raw",
                )
            try:
                parsed = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                return _fail("arguments_raw", f"turn {i} arguments_raw is not valid JSON: {exc}")
            if parsed != call.get("arguments"):
                return _fail(
                    "arguments_raw",
                    f"turn {i} tool call {call.get('id')!r}: arguments_raw != arguments",
                )
    return _ok("arguments_raw")


def check_tool_names(record: dict, known_tools: Iterable[str] | None = None) -> CheckResult:
    """Tool names must be in the set the harness advertised (when known)."""
    if known_tools is None:
        return _ok("tool_names", note="no advertised tool set available")
    known_set = set(known_tools)
    if not known_set:
        return _ok("tool_names", note="advertised tool set was empty")
    seen: set[str] = set()
    for turn in record.get("trajectory", {}).get("turns", []):
        for call in turn.get("tool_calls", []) or []:
            name = call.get("name")
            if name not in known_set:
                return _fail("tool_names", f"tool {name!r} was not advertised", tool=name)
            seen.add(name)
    return _ok("tool_names", tools_used=len(seen))


def check_no_truncated_output(record: dict) -> CheckResult:
    """Reject any trajectory containing a truncated tool result.

    A truncated tool result teaches the model to *invent* the rest of the file.
    This is the single most destructive silent poison in agentic datasets.
    """
    turns = record.get("trajectory", {}).get("turns", [])
    for i, turn in enumerate(turns):
        if turn.get("truncated"):
            return _fail(
                "no_truncated_output",
                f"turn {i} has a truncated tool result",
                tool=turn.get("name"),
            )
    return _ok("no_truncated_output")


def check_no_secrets(record: dict) -> CheckResult:
    """Scan the whole record for credential-shaped strings."""
    blob = json.dumps(record, ensure_ascii=False)
    for regex, name in _SECRET_RES:
        match = regex.search(blob)
        if match:
            snippet = match.group(0)
            if _PLACEHOLDERS.search(snippet):
                continue
            return _fail("no_secrets", f"credential-shaped string detected ({name})", kind=name)
    return _ok("no_secrets")


def check_no_tool_loop(record: dict, max_identical: int = 3) -> CheckResult:
    """Reject trajectories that repeat the identical tool call too many times."""
    counts: dict[str, int] = {}
    for turn in record.get("trajectory", {}).get("turns", []):
        for call in turn.get("tool_calls", []) or []:
            try:
                args = json.dumps(call.get("arguments"), sort_keys=True, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                args = str(call.get("arguments"))
            key = f"{call.get('name')}::{args}"
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return _ok("no_tool_loop")
    worst_key, worst = max(counts.items(), key=lambda kv: kv[1])
    if worst >= max_identical:
        return _fail(
            "no_tool_loop",
            f"identical tool call repeated {worst} times (limit {max_identical})",
            tool=worst_key.split("::", 1)[0],
            repeats=worst,
        )
    return _ok("no_tool_loop", max_repeats=worst)


def check_length_bounds(record: dict, min_chars: int = 400, max_chars: int = 400_000) -> CheckResult:
    """Bound the trajectory size. Too short teaches nothing; too long blows the context."""
    turns = record.get("trajectory", {}).get("turns", [])
    total = sum(len(t.get("content") or "") for t in turns)
    if total < min_chars:
        return _fail("length_bounds", f"trajectory is only {total} chars", chars=total)
    if total > max_chars:
        return _fail("length_bounds", f"trajectory is {total} chars (limit {max_chars})", chars=total)
    return _ok("length_bounds", chars=total)


def check_provenance(record: dict) -> CheckResult:
    """The licence gate. ``trainable`` is a separate permission from ``license``."""
    prov = record.get("provenance", {})
    if not prov.get("license_verified"):
        return _fail("provenance", "license was not verified")
    if not prov.get("trainable"):
        return _fail("provenance", "source is not marked trainable")
    if not prov.get("source_id"):
        return _fail("provenance", "no source_id")
    if not prov.get("collected_at"):
        return _fail("provenance", "no collected_at timestamp")
    return _ok("provenance", source_id=prov.get("source_id"))


def check_verification_consistency(record: dict) -> CheckResult:
    """A row claiming ``pass`` must have the evidence to back it up."""
    ver = record.get("verification", {})
    status = ver.get("status")
    if status == "pass":
        env = record.get("environment", {})
        if not env.get("fail_to_pass"):
            return _fail("verification_consistency", "status=pass but fail_to_pass is empty")
        tests = ver.get("tests", {}).get("pass_to_pass", {})
        if int(tests.get("regressed", 0) or 0) > 0:
            return _fail("verification_consistency", f"{tests['regressed']} regression(s)")
        if ver.get("regressions"):
            return _fail("verification_consistency", "verification lists regressions")
    elif status == "skipped":
        # Allowed, but it must not be presented as verified.
        if record.get("quality", {}).get("verified") is True:
            return _fail("verification_consistency", "quality.verified is true but verification was skipped")
    return _ok("verification_consistency", status=status)


def check_no_hallucinated_paths(record: dict, repo_files: set[str] | None = None) -> CheckResult:
    """Edited paths must exist in the repository (only checkable when the repo is known)."""
    if not repo_files:
        return _ok("no_hallucinated_paths", note="repo file list not available")
    bad: list[str] = []
    for patch in record.get("trajectory", {}).get("patches", []) or []:
        path = patch.get("path")
        if path and path not in repo_files:
            bad.append(path)
    if bad:
        return _fail("no_hallucinated_paths", f"{len(bad)} patched path(s) not in repo", paths=bad[:10])
    return _ok("no_hallucinated_paths")


def check_honest_conclusion(record: dict) -> CheckResult:
    """Reject a trajectory that claims success while the verification says otherwise.

    Note this is compatible with keeping ``failed_but_honest`` examples: those are
    rows where the agent *did not* claim success. A row that claims success on a
    failed run is dishonest and must not enter the dataset.
    """
    ver = record.get("verification", {})
    if ver.get("status") == "pass":
        return _ok("honest_conclusion")
    turns = record.get("trajectory", {}).get("turns", [])
    final_text = ""
    for turn in reversed(turns):
        if turn.get("role") == "assistant" and (turn.get("content") or "").strip():
            final_text = turn["content"]
            break
    if final_text and _SUCCESS_CLAIM.search(final_text) and not _NO_SUCCESS_CLAIM.search(final_text):
        return _fail(
            "honest_conclusion",
            "final message claims success but verification did not pass",
            status=ver.get("status"),
        )
    return _ok("honest_conclusion", status=ver.get("status"))


# --------------------------------------------------------------------------- runner
def run_tier_a(
    record: dict,
    validator: Any | None = None,
    known_tools: Iterable[str] | None = None,
    repo_files: set[str] | None = None,
    min_chars: int = 400,
    max_chars: int = 400_000,
    max_identical_tool_calls: int = 3,
) -> list[CheckResult]:
    """Run every Tier A check. Returns one result per check, in a stable order."""
    return [
        check_schema(record, validator),
        check_turn_sequence(record),
        check_tool_call_pairing(record),
        check_arguments_raw_agreement(record),
        check_tool_names(record, known_tools),
        check_no_truncated_output(record),
        check_no_secrets(record),
        check_no_tool_loop(record, max_identical_tool_calls),
        check_length_bounds(record, min_chars, max_chars),
        check_provenance(record),
        check_verification_consistency(record),
        check_no_hallucinated_paths(record, repo_files),
        check_honest_conclusion(record),
    ]
