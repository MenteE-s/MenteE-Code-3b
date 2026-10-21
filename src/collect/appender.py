"""Append-only JSONL writer with crash-safety and write-integrity guarantees.

Design constraints (see docs/05-DATA-COLLECTION-PLAYBOOK.md and the plan):

* **Never truncates.** The file is opened in binary *append* mode only. There is no
  code path in this module that can remove existing rows.
* **Never leaves a partial line.** Each record is serialised and written as a single
  complete bytes payload in one ``write()`` call. On open, a trailing partial line
  (from a process killed mid-write) is healed by truncating back to the last newline.
* **Never loses acknowledged rows.** ``flush()`` after every record, ``fsync()``
  every N records and at every checkpoint.
* **Never silently writes nothing.** The writer keeps its own belief about how many
  lines/bytes it has written. :func:`src.collect.integrity.verify` compares that
  belief against the file on disk and aborts the run on mismatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["JsonlAppender", "AppendError", "dumps"]


class AppendError(RuntimeError):
    """Raised when the appender cannot guarantee a safe write."""


def dumps(obj: Any) -> str:
    """Compact, UTF-8, key-order-preserving JSON.

    Key order is *not* sorted: the schema defines the canonical field order and
    keeping it makes the file diffable and reviewable by a human.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _count_lines_and_bytes(path: Path) -> tuple[int, int]:
    """Count newlines and bytes without loading the file into memory."""
    if not path.exists():
        return 0, 0
    total = 0
    lines = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            lines += chunk.count(b"\n")
    return lines, total


class JsonlAppender:
    """Append-only JSONL writer.

    Parameters
    ----------
    path:
        Target file. Parent directories are created.
    fsync_every:
        ``fsync`` the file every N appended records. ``1`` disables batching.
    lock:
        Take an advisory lock on ``<path>.lock`` so two collectors cannot write the
        same file concurrently. Interleaved writers are the one failure mode that
        append-mode cannot protect against on its own.
    """

    def __init__(self, path: str | os.PathLike[str], fsync_every: int = 10, lock: bool = True):
        self.path = Path(path)
        self.fsync_every = max(1, int(fsync_every))
        self.use_lock = lock
        self._fh = None
        self._lock_fh = None
        self.lines = 0
        self.bytes = 0
        self.healed_bytes = 0

    # ------------------------------------------------------------------ lifecycle
    def open(self, resume: bool = True) -> "JsonlAppender":
        """Open the file for appending.

        ``resume=False`` refuses to open a file that already has content — this is
        the guard against a second run silently truncating hours of work.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_lock:
            self._acquire_lock()

        existed = self.path.exists() and self.path.stat().st_size > 0
        if existed and not resume:
            self.close()
            raise AppendError(
                f"{self.path} already contains data. Refusing to open without "
                f"resume=True (this writer never truncates by design)."
            )

        self.healed_bytes = self._heal_tail()
        # 'ab' is the ONLY mode used for writing. 'w'/'wb' would truncate.
        self._fh = open(self.path, "ab")
        # Recount from disk so the counters reflect reality, not belief.
        self.lines, self.bytes = _count_lines_and_bytes(self.path)
        return self

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        self._release_lock()

    def __enter__(self) -> "JsonlAppender":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writing
    def append(self, obj: Any) -> int:
        """Append one record. Returns the number of bytes written."""
        if self._fh is None:
            raise AppendError("appender is not open")
        payload = (dumps(obj) + "\n").encode("utf-8")
        # One write() of a complete payload == one complete line.
        self._fh.write(payload)
        self._fh.flush()
        self.lines += 1
        self.bytes += len(payload)
        if self.lines % self.fsync_every == 0:
            os.fsync(self._fh.fileno())
        return len(payload)

    def checkpoint(self) -> None:
        """Force durability now. Called at every integrity checkpoint."""
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    # ------------------------------------------------------------------ helpers
    def _heal_tail(self) -> int:
        """Remove a trailing partial line left by a killed process.

        Returns the number of bytes discarded.
        """
        if not self.path.exists():
            return 0
        size = self.path.stat().st_size
        if size == 0:
            return 0
        with open(self.path, "rb") as fh:
            fh.seek(size - 1)
            if fh.read(1) == b"\n":
                return 0
            # Walk backwards looking for the last newline.
            pos = size
            window = 1 << 20
            while pos > 0:
                start = max(0, pos - window)
                fh.seek(start)
                buf = fh.read(pos - start)
                idx = buf.rfind(b"\n")
                if idx != -1:
                    keep = start + idx + 1
                    with open(self.path, "r+b") as w:
                        w.truncate(keep)
                        w.flush()
                        os.fsync(w.fileno())
                    return size - keep
                pos = start
            # No newline anywhere: the whole file is a single partial record.
            with open(self.path, "r+b") as w:
                w.truncate(0)
                w.flush()
                os.fsync(w.fileno())
            return size

    def _acquire_lock(self) -> None:
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fh = open(lock_path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_fh.seek(0)
                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_fh.close()
            self._lock_fh = None
            raise AppendError(
                f"another collector appears to hold the lock on {self.path}: {exc}"
            ) from exc

    def _release_lock(self) -> None:
        if self._lock_fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_fh.seek(0)
                msvcrt.locking(self._lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._lock_fh.close()
        except OSError:
            pass
        self._lock_fh = None
