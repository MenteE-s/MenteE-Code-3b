"""Merge several collectors' datasets into one, safely.

Why this is not just concatenation
----------------------------------
Four people running the collector on four machines produce four files that must
combine into one clean dataset. Three things go wrong if you just ``cat`` them:

1. **Duplicate trajectory ids** - if any collector ran without a collector id,
   its ``traj_2026-10-23_000001`` collides with everyone else's.
2. **Duplicate content** - two collectors attempting the *same task* with the same
   model produce near-identical trajectories. Training on both is the same as
   training on one, but it silently halves your task diversity.
3. **One bad machine poisons the set** - a collector with a broken sandbox
   produces rows that look fine and are garbage. You want that visible *per
   collector*, before the merge, not discovered at training time.

So the merge: verifies every input first, schema-validates every row, drops
duplicate ids and duplicate content, and reports per-collector statistics.

Dedup policy
------------
* same ``id``                      -> hard duplicate, dropped
* same content key                 -> hard duplicate, dropped
* same task, different content     -> **kept** (distinct attempts are real
  training signal, and the minhash near-dedup pass catches true near-dupes later)

The content key ignores the id, timestamps and collector identity, so two
collectors producing the identical attempt collapse into one row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .appender import JsonlAppender, dumps
from .integrity import verify

__all__ = ["MergeSourceStats", "MergeReport", "merge_datasets", "content_key"]


@dataclass
class MergeSourceStats:
    path: str
    rows: int = 0
    accepted: int = 0
    duplicate_id: int = 0
    duplicate_content: int = 0
    schema_invalid: int = 0
    integrity_ok: bool = True
    integrity_note: str = ""
    verified: int = 0
    failure_path: int = 0


@dataclass
class MergeReport:
    ok: bool = True
    sources: list[MergeSourceStats] = field(default_factory=list)
    total_accepted: int = 0
    total_duplicate_id: int = 0
    total_duplicate_content: int = 0
    total_schema_invalid: int = 0
    output_path: str = ""
    output_bytes: int = 0
    output_lines: int = 0
    notes: list[str] = field(default_factory=list)


def content_key(record: dict[str, Any]) -> str:
    """A hash of everything that makes an attempt *the same attempt*.

    Deliberately excludes the id, timestamps and collector identity: two
    collectors producing the identical attempt should collapse into one row.
    """
    payload = {
        "instruction": record.get("task", {}).get("instruction", ""),
        "base": record.get("environment", {}).get("base_commit", ""),
        "repo": record.get("environment", {}).get("repo", ""),
        "turns": record.get("trajectory", {}).get("turns", []),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical(record: dict[str, Any]) -> dict[str, Any]:
    """Strip volatile fields so comparisons and the merged file are stable."""
    out = json.loads(json.dumps(record, ensure_ascii=False))
    prov = out.get("provenance", {})
    if isinstance(prov, dict):
        prov.pop("collected_at", None)
        prov["collected_by"] = prov.get("source_id", "merged")
    return out


def merge_datasets(
    inputs: Iterable[str | Path],
    out_path: str | Path,
    validator: Any | None = None,
    require_integrity: bool = True,
    verbose: bool = True,
) -> MergeReport:
    """Merge many collector datasets into one verified file."""
    report = MergeReport()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    seen_content: set[str] = set()

    appender = JsonlAppender(out_path, fsync_every=25)
    appender.open(resume=False)  # a merged file is always written fresh

    try:
        for src in inputs:
            p = Path(src)
            st = MergeSourceStats(path=str(p))
            report.sources.append(st)

            if not p.exists() or p.stat().st_size == 0:
                st.integrity_note = "missing or empty"
                st.integrity_ok = False
                if require_integrity:
                    report.ok = False
                continue

            # integrity first: never merge from a file that fails its own check
            pre = verify(p, deep=True, schema=None)
            st.integrity_ok = pre.ok
            st.integrity_note = pre.summary()
            if not pre.ok and require_integrity:
                report.ok = False
                if verbose:
                    print(f"  ! skipping {p.name}: {pre.summary()}")
                continue

            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                st.rows += 1
                try:
                    record = json.loads(line)
                except Exception:  # noqa: BLE001
                    st.schema_invalid += 1
                    continue
                if not isinstance(record, dict):
                    st.schema_invalid += 1
                    continue

                if validator is not None:
                    errs = next(iter(validator.iter_errors(record)), None)
                    if errs is not None:
                        st.schema_invalid += 1
                        continue

                rid = record.get("id", "")
                ckey = content_key(record)

                if rid in seen_ids:
                    st.duplicate_id += 1
                    continue
                if ckey in seen_content:
                    st.duplicate_content += 1
                    continue

                seen_ids.add(rid)
                seen_content.add(ckey)
                clean = _canonical(record)
                appender.append(clean)
                st.accepted += 1
                if record.get("quality", {}).get("verified"):
                    st.verified += 1
                if record.get("quality", {}).get("checks", {}).get("has_failure_path"):
                    st.failure_path += 1
            if verbose:
                print(
                    f"  {p.name:<34} rows={st.rows:<6} kept={st.accepted:<6} "
                    f"dup_id={st.duplicate_id:<4} dup_content={st.duplicate_content:<4} "
                    f"invalid={st.schema_invalid}"
                )
    finally:
        appender.close()

    report.total_accepted = sum(s.accepted for s in report.sources)
    report.total_duplicate_id = sum(s.duplicate_id for s in report.sources)
    report.total_duplicate_content = sum(s.duplicate_content for s in report.sources)
    report.total_schema_invalid = sum(s.schema_invalid for s in report.sources)
    report.output_path = str(out_path)
    report.output_bytes = out_path.stat().st_size if out_path.exists() else 0
    report.output_lines = appender.lines

    # the merged file must itself be verified before we call it done
    post = verify(out_path, believed_lines=appender.lines, believed_bytes=appender.bytes, deep=True)
    if not post.ok:
        report.ok = False
        report.notes.append(f"merged file failed verification: {post.summary()}")

    if report.total_accepted == 0:
        report.notes.append("no rows accepted - nothing to merge?")
    return report
