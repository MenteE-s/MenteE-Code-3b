"""Running statistics and reporting.

The report is the artefact you read at the end of a session to answer "is this
data any good, and how much did it cost?". It is written to disk as JSON so the
same numbers feed a run report, a dataset quality report, and the model card.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = ["Reporter"]


class Reporter:
    """Accumulates per-run statistics."""

    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        self.started = time.time()
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.skipped = 0
        self.deep_checked = 0
        self.rejections: Counter[str] = Counter()
        self.rejection_detail: Counter[str] = Counter()
        self.accepted_tokens = 0
        self.verification_status: Counter[str] = Counter()
        self.turns_total = 0
        self.turns_max = 0
        self.failure_path_count = 0
        self.integrity_checks = 0
        self.integrity_failures = 0
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ events
    def accept(self, record: dict[str, Any]) -> None:
        self.accepted += 1
        traj = record.get("trajectory", {})
        n = int(traj.get("n_turns", 0) or 0)
        self.turns_total += n
        self.turns_max = max(self.turns_max, n)
        tokens = traj.get("total_tokens", {}) or {}
        self.accepted_tokens += int(tokens.get("input", 0) or 0) + int(tokens.get("output", 0) or 0)
        status = record.get("verification", {}).get("status", "unknown")
        self.verification_status[status] += 1
        checks = record.get("quality", {}).get("checks", {}) or {}
        if checks.get("has_failure_path"):
            self.failure_path_count += 1

    def reject(self, reasons: list[str]) -> None:
        self.rejected += 1
        for reason in reasons:
            name = reason.split(":", 1)[0].strip()
            self.rejections[name] += 1
            self.rejection_detail[reason[:200]] += 1

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.rejections[reason] += 1

    def note(self, kind: str, detail: str) -> None:
        self.events.append({"at": time.strftime("%H:%M:%S"), "kind": kind, "detail": detail})

    # ------------------------------------------------------------------ report
    @property
    def acceptance_rate(self) -> float:
        total = self.accepted + self.rejected
        return (self.accepted / total) if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        accepted = max(1, self.accepted)
        return {
            "run_id": self.run_id,
            "elapsed_s": round(time.time() - self.started, 1),
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "deep_checked": self.deep_checked,
            "integrity_checks": self.integrity_checks,
            "integrity_failures": self.integrity_failures,
            "verification_status": dict(self.verification_status),
            "verified_rate": round(self.verification_status.get("pass", 0) / accepted, 4),
            "failure_path_rate": round(self.failure_path_count / accepted, 4),
            "mean_turns": round(self.turns_total / accepted, 2),
            "max_turns": self.turns_max,
            "mean_tokens": round(self.accepted_tokens / accepted, 1),
            "rejections_by_check": dict(self.rejections.most_common()),
            "rejection_examples": dict(self.rejection_detail.most_common(20)),
            "events": self.events[-50:],
        }

    def render(self) -> str:
        s = self.snapshot()
        lines = [
            "",
            "  Collection summary",
            "  ------------------",
            f"  generated            {s['generated']}",
            f"  accepted             {s['accepted']}   ({s['acceptance_rate']*100:.1f}%)",
            f"  rejected             {s['rejected']}",
            f"  skipped              {s['skipped']}",
            f"  verified rate        {s['verified_rate']*100:.1f}%",
            f"  failure-path rate    {s['failure_path_rate']*100:.1f}%   <- target >= 30%",
            f"  mean turns           {s['mean_turns']}",
            f"  mean tokens          {s['mean_tokens']}",
            f"  integrity checks     {s['integrity_checks']} ({s['integrity_failures']} failed)",
            f"  elapsed              {s['elapsed_s']}s",
        ]
        if s["rejections_by_check"]:
            lines.append("")
            lines.append("  rejections by check:")
            for name, count in s["rejections_by_check"].items():
                lines.append(f"    {name:<32} {count}")
        return "\n".join(lines)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")
