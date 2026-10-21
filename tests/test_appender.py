"""Appender tests: the guarantees that make a long unattended run safe.

The behaviours under test are the ones that fail *quietly* in a naive
implementation:

* a second run must not truncate existing rows;
* a process killed mid-write must not leave a half-line that breaks the file;
* the writer's belief about what it wrote must match the file on disk;
* a crash must not lose rows that were acknowledged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.collect.appender import AppendError, JsonlAppender
from src.collect.integrity import verify


def _row(i: int) -> dict:
    return {
        "schema_version": "1.0.0",
        "id": f"traj_2026-10-21_{i:06d}",
        "provenance": {"source_id": "t"},
        "payload": "x" * 50,
    }


# --------------------------------------------------------------------- basic
def test_append_creates_and_counts(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target, fsync_every=1) as app:
        for i in range(5):
            app.append(_row(i))
    assert app.lines == 5
    assert target.read_text(encoding="utf-8").count("\n") == 5
    report = verify(target, believed_lines=5, believed_bytes=app.bytes)
    assert report.ok, report.summary()


def test_append_is_append_only(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        for i in range(3):
            app.append(_row(i))
    with JsonlAppender(target) as app:
        for i in range(3, 6):
            app.append(_row(i))
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert json.loads(lines[0])["id"] == "traj_2026-10-21_000000"
    assert json.loads(lines[-1])["id"] == "traj_2026-10-21_000005"


def test_refuses_to_open_existing_without_resume(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
    with pytest.raises(AppendError):
        JsonlAppender(target).open(resume=False)


def test_resume_continues_counters(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
        app.append(_row(1))
    with JsonlAppender(target) as app:
        assert app.lines == 2
        app.append(_row(2))
    assert len(target.read_text(encoding="utf-8").splitlines()) == 3


# --------------------------------------------------------------------- healing
def test_heals_partial_trailing_line(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
        app.append(_row(1))
    # Simulate a process killed mid-write.
    with open(target, "ab") as fh:
        fh.write(b'{"schema_version":"1.0.0","id":"traj_2026-10-2')
    assert not target.read_bytes().endswith(b"\n")

    with JsonlAppender(target) as app:
        assert app.healed_bytes > 0
        assert app.lines == 2
        app.append(_row(2))

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # must all parse


def test_heals_file_that_is_entirely_partial(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    target.write_bytes(b'{"id":"broken')
    with JsonlAppender(target) as app:
        assert app.healed_bytes == len(b'{"id":"broken')
        assert app.lines == 0
        app.append(_row(0))
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1


# --------------------------------------------------------------------- integrity
def test_belief_mismatch_is_detected(tmp_path: Path):
    """The failure mode this whole module exists to catch."""
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
        app.append(_row(1))
    report = verify(target, believed_lines=500, believed_bytes=app.bytes)
    assert not report.ok
    assert any("line count mismatch" in n for n in report.notes)


def test_bytes_mismatch_is_detected(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
    report = verify(target, believed_lines=app.lines, believed_bytes=app.bytes + 999)
    assert not report.ok
    assert any("byte count mismatch" in n for n in report.notes)


def test_verify_detects_corrupt_line(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
    with open(target, "ab") as fh:
        fh.write(b"{not json}\n")
    report = verify(target, deep=True)
    assert not report.ok
    assert report.parse_errors


def test_verify_detects_duplicate_ids(tmp_path: Path):
    target = tmp_path / "train.jsonl"
    with JsonlAppender(target) as app:
        app.append(_row(0))
        app.append(_row(0))
    report = verify(target, deep=True)
    assert not report.ok
    assert report.duplicate_ids


# --------------------------------------------------------------------- crash
CRASH_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, {repo!r})
    from src.collect.appender import JsonlAppender
    target = {target!r}
    app = JsonlAppender(target, fsync_every=1).open()
    for i in range(200):
        app.append({{"id": "row-%d" % i, "blob": "y" * 200}})
        if i == 120:
            # simulate a hard kill: write a partial record then die
            with open(target, "ab") as fh:
                fh.write(b'{{"id":"row-121-part')
            os._exit(9)
    """
)


def test_survives_a_hard_kill(tmp_path: Path):
    """Kill a writer mid-record; the next open must heal and the file must parse."""
    target = tmp_path / "train.jsonl"
    repo = str(Path(__file__).resolve().parents[1])
    script = CRASH_SCRIPT.format(repo=repo, target=str(target))

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert proc.returncode == 9, proc.stderr.decode(errors="replace")

    # The file on disk must currently be broken...
    raw = target.read_bytes()
    assert not raw.endswith(b"\n")

    # ...and the next writer must heal it and produce a fully valid file.
    with JsonlAppender(target, fsync_every=1) as app:
        assert app.healed_bytes > 0
        app.append({"id": "after-crash"})

    report = verify(target, deep=True)
    assert report.ok, report.summary()
    ids = [json.loads(line)["id"] for line in target.read_text(encoding="utf-8").splitlines()]
    assert ids[-1] == "after-crash"
    assert "row-120" in ids
