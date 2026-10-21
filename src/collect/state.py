"""Sidecar state: the collector's belief about what it has written.

Why this exists
---------------
The appender can *silently* fail to write (wrong path, permissions, a crash before
flush). On its own, nothing notices — the run reports success and the file is empty
or short. The sidecar stores the counters **before** they are trusted, so
:func:`src.collect.integrity.verify` can compare belief against disk and abort.

Files written next to the dataset
---------------------------------
``train.jsonl``            the data
``train.jsonl.state.json`` believed counters + run metadata
``train.jsonl.ids``        one trajectory id per line (duplicate detection)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = ["Sidecar"]


class Sidecar:
    """Persistent state for a collector run."""

    def __init__(self, data_path: str | os.PathLike[str]):
        self.data_path = Path(data_path)
        self.state_path = self.data_path.with_name(self.data_path.name + ".state.json")
        self.ids_path = self.data_path.with_name(self.data_path.name + ".ids")
        self.data: dict[str, Any] = {}
        self._ids: set[str] = set()
        self._ids_fh = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> "Sidecar":
        self.data = {}
        if self.state_path.exists():
            try:
                self.data = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                # A corrupt sidecar must not be silently ignored: it holds the
                # counters we compare against. Start clean but say so loudly.
                self.data = {"_state_corrupt": True}
        self._ids = set()
        if self.ids_path.exists():
            for line in self.ids_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._ids.add(line)
        return self

    def open(self) -> "Sidecar":
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)
        self._ids_fh = open(self.ids_path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._ids_fh is not None:
            try:
                self._ids_fh.flush()
                os.fsync(self._ids_fh.fileno())
                self._ids_fh.close()
            except (OSError, ValueError):
                pass
            self._ids_fh = None

    def __enter__(self) -> "Sidecar":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ counters
    @property
    def lines(self) -> int:
        return int(self.data.get("lines", 0))

    @property
    def bytes(self) -> int:
        return int(self.data.get("bytes", 0))

    @property
    def run_id(self) -> str:
        return str(self.data.get("run_id", ""))

    # ------------------------------------------------------------------ mutation
    def has_id(self, trajectory_id: str) -> bool:
        return trajectory_id in self._ids

    def add_id(self, trajectory_id: str) -> None:
        if trajectory_id in self._ids:
            return
        self._ids.add(trajectory_id)
        if self._ids_fh is not None:
            self._ids_fh.write(trajectory_id + "\n")
            self._ids_fh.flush()

    def set_run(self, run_id: str, extra: dict | None = None) -> None:
        self.data["run_id"] = run_id
        self.data["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if extra:
            self.data.update(extra)

    def checkpoint(
        self,
        lines: int,
        bytes_written: int,
        last_id: str | None = None,
        rejections: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        """Persist the belief counters. Called at every integrity checkpoint."""
        self.data["lines"] = int(lines)
        self.data["bytes"] = int(bytes_written)
        if last_id is not None:
            self.data["last_id"] = last_id
        self.data["checkpointed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if rejections is not None:
            self.data["rejections"] = rejections
        if extra:
            self.data.update(extra)
        self.save()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Atomic replace so a crash mid-write cannot corrupt the sidecar.
        os.replace(tmp, self.state_path)
