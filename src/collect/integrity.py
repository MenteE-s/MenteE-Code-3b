"""Write-integrity verification for append-only JSONL datasets.

The failure this module exists to catch: **the collector believes it wrote 4,000 rows
but the file on disk contains 0 bytes** (wrong path, permissions, a buffering bug, a
process that crashed before flushing). Nothing else in the pipeline would notice.

The contract
------------
* :func:`verify` compares the *believed* counters (from the sidecar state file) with
  the *actual* counters read from disk.
* A mismatch is a hard failure. The collector aborts rather than appending to a file
  that is not keeping up.
* Cheap checks (line count, byte count, trailing newline) run on every checkpoint.
* The full re-parse (every line is parsed and schema-validated) runs on a schedule
  that widens over time, and always on resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = ["IntegrityReport", "verify", "IntegritySchedule"]


@dataclass
class IntegrityReport:
    """Result of an integrity check."""

    path: str
    ok: bool = True
    disk_bytes: int = 0
    disk_lines: int = 0
    believed_lines: int | None = None
    believed_bytes: int | None = None
    ends_with_newline: bool = True
    parse_errors: list[tuple[int, str]] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    schema_errors: list[tuple[int, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.notes.append(reason)

    def summary(self) -> str:
        head = "OK" if self.ok else "FAIL"
        parts = [
            f"[{head}] {self.path}",
            f"lines={self.disk_lines}",
            f"bytes={self.disk_bytes}",
        ]
        if self.believed_lines is not None:
            parts.append(f"believed_lines={self.believed_lines}")
            parts.append(f"believed_bytes={self.believed_bytes}")
        if not self.ends_with_newline:
            parts.append("TAIL_NOT_NEWLINE")
        if self.parse_errors:
            parts.append(f"parse_errors={len(self.parse_errors)}")
        if self.schema_errors:
            parts.append(f"schema_errors={len(self.schema_errors)}")
        if self.duplicate_ids:
            parts.append(f"duplicate_ids={len(self.duplicate_ids)}")
        for note in self.notes:
            parts.append(f"! {note}")
        return " ".join(parts)


def verify(
    path: str | Path,
    believed_lines: int | None = None,
    believed_bytes: int | None = None,
    deep: bool = False,
    schema: dict | None = None,
    validator: Any | None = None,
    tail_window: int = 200,
) -> IntegrityReport:
    """Check that the file on disk matches what we believe we wrote.

    Parameters
    ----------
    deep:
        Parse and schema-validate **every** line. Off by default because it is O(n).
    schema / validator:
        An optional JSON Schema (and a pre-built ``jsonschema`` validator) used for
        per-line validation during a deep check.
    tail_window:
        When ``deep`` is False, only the last N lines are parsed. That still catches
        corruption near the write frontier, which is where it actually happens.
    """
    p = Path(path)
    report = IntegrityReport(path=str(p))

    if not p.exists():
        report.disk_bytes = 0
        report.disk_lines = 0
        if believed_lines:
            report.fail("file does not exist but rows were believed to be written")
        else:
            report.notes.append("file does not exist yet (nothing written)")
        report.ends_with_newline = True
        return report

    data = p.read_bytes()
    report.disk_bytes = len(data)
    report.believed_lines = believed_lines
    report.believed_bytes = believed_bytes
    report.ends_with_newline = (not data) or data.endswith(b"\n")

    if data and not report.ends_with_newline:
        report.fail("file does not end with a newline (partial trailing record)")

    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    report.disk_lines = len(lines)

    if believed_lines is not None and report.disk_lines != believed_lines:
        report.fail(
            f"line count mismatch: disk has {report.disk_lines}, "
            f"we believed {believed_lines} (delta {report.disk_lines - believed_lines})"
        )
    if believed_bytes is not None and report.disk_bytes != believed_bytes:
        report.fail(
            f"byte count mismatch: disk has {report.disk_bytes}, "
            f"we believed {believed_bytes} (delta {report.disk_bytes - believed_bytes})"
        )

    to_parse: Iterable[tuple[int, bytes]]
    if deep:
        to_parse = enumerate(lines)
    else:
        start = max(0, len(lines) - tail_window)
        to_parse = enumerate(lines[start:], start=start)

    seen_ids: dict[str, int] = {}
    for idx, raw in to_parse:
        if not raw.strip():
            report.parse_errors.append((idx + 1, "blank line"))
            continue
        try:
            obj = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            report.parse_errors.append((idx + 1, f"{type(exc).__name__}: {exc}"))
            continue
        if not isinstance(obj, dict):
            report.parse_errors.append((idx + 1, "record is not a JSON object"))
            continue
        rid = obj.get("id")
        if isinstance(rid, str):
            if rid in seen_ids:
                report.duplicate_ids.append(rid)
            else:
                seen_ids[rid] = idx + 1
        if schema is not None and validator is not None:
            for err in validator.iter_errors(obj):
                report.schema_errors.append((idx + 1, err.message))
                break

    if report.parse_errors:
        report.fail(f"{len(report.parse_errors)} record(s) failed to parse")
    if report.schema_errors:
        report.fail(f"{len(report.schema_errors)} record(s) failed schema validation")
    if report.duplicate_ids:
        report.fail(f"{len(report.duplicate_ids)} duplicate id(s)")

    return report


class IntegritySchedule:
    """Widening check schedule with jitter.

    The first checks are frequent (cheap, catches a broken appender in the first 10
    rows). The gaps widen as the file grows, and jitter keeps the check from
    landing on a predictable beat — a check that always lands *between* writes
    proves nothing.

    Example with the defaults: 10, ~40, ~85, ~145, ~235, then every ~200.
    """

    def __init__(
        self,
        first_check_at: int = 10,
        gaps: Iterable[int] = (30, 45, 60, 90),
        steady_every: int = 200,
        jitter_ratio: float = 0.30,
        full_rescan_every: int = 5,
        seed: int = 0,
    ):
        self.first_check_at = max(1, int(first_check_at))
        self.gaps = [int(g) for g in gaps]
        self.steady_every = max(1, int(steady_every))
        self.jitter_ratio = max(0.0, float(jitter_ratio))
        self.full_rescan_every = max(1, int(full_rescan_every))
        self.seed = seed
        self._thresholds: list[int] = []
        self._build()
        self._checks = 0

    def _build(self) -> None:
        import random

        rng = random.Random(self.seed)
        n = self.first_check_at
        self._thresholds.append(n)
        for gap in self.gaps:
            n += max(1, int(gap * (1 + rng.uniform(-self.jitter_ratio, self.jitter_ratio))))
            self._thresholds.append(n)
        while n < 10**9:
            n += max(1, int(self.steady_every * (1 + rng.uniform(-self.jitter_ratio, self.jitter_ratio))))
            self._thresholds.append(n)

    def due(self, lines: int) -> bool:
        """True when a check should run at this line count."""
        if not self._thresholds:
            return False
        return lines >= self._thresholds[0]

    def next_threshold(self) -> int:
        return self._thresholds[0] if self._thresholds else -1

    def mark_checked(self) -> None:
        """Advance the schedule after a check has run."""
        self._checks += 1
        if self._thresholds:
            self._thresholds.pop(0)

    def is_deep(self) -> bool:
        """True when this check should re-parse every line."""
        return self._checks % self.full_rescan_every == 0

    def plan(self) -> list[int]:
        return list(self._thresholds)
