"""End-to-end smoke test of the collector loop.

Generates a synthetic source, runs the real appender / validator / integrity
machinery over it, and asserts the file on disk is exactly what we think it is.

This is the "does the whole loop actually work" test, and it exercises the
append-integrity guarantees on a real file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collect.appender import JsonlAppender
from src.collect.integrity import IntegritySchedule, verify
from src.collect.normalizer import CommonMessage, RawTrajectory, build_record
from src.collect.report import Reporter
from src.collect.state import Sidecar
from src.collect.validate import TierAValidator, load_schema, make_validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
VALIDATOR = make_validator(SCHEMA)

LONG_BODY = (
    "def parse_window(tokens, size):\n"
    "    if not tokens:\n"
    "        return tokens.index('[')\n"
    "    start = tokens.index('[')\n"
    "    end = tokens.index(']')\n"
    "    body = tokens[start + 1:end]\n"
    "    return body[0:size - 1]\n"
)


def synthetic_trajectory(i: int) -> RawTrajectory:
    messages = [
        CommonMessage(role="system", content="You are a coding agent working in a Python repo."),
        CommonMessage(role="user", content=f"Fix the off-by-one in parse_window (case {i})."),
        CommonMessage(
            role="assistant",
            content="Reading the parser module first.",
            tool_calls=[
                {
                    "id": "call_0",
                    "name": "read_file",
                    "arguments": {"path": "src/parser.py"},
                    "arguments_raw": '{"path":"src/parser.py"}',
                }
            ],
        ),
        CommonMessage(
            role="tool_result",
            content=LONG_BODY,
            tool_results=[
                {
                    "tool_call_id": "call_0",
                    "name": "read_file",
                    "content": LONG_BODY,
                    "truncated": False,
                }
            ],
        ),
        CommonMessage(
            role="assistant",
            content="The guard returns tokens.index('[') which raises on an empty list. Patching.",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "apply_patch",
                    "arguments": {"path": "src/parser.py"},
                    "arguments_raw": '{"path":"src/parser.py"}',
                }
            ],
        ),
        CommonMessage(
            role="tool_result",
            content="Applied patch to src/parser.py (+1 -1)",
            tool_results=[
                {
                    "tool_call_id": "call_1",
                    "name": "apply_patch",
                    "content": "Applied patch to src/parser.py (+1 -1)",
                    "truncated": False,
                }
            ],
        ),
        CommonMessage(role="assistant", content="The guard is fixed and the full test suite passes."),
    ]
    return RawTrajectory(
        task={
            "instruction": f"Fix the off-by-one in parse_window (case {i}).",
            "repo": "acme/windowlib",
            "base_commit": "9f2c1ab",
            "language": "python",
            "task_type": "bug_fix",
            "fail_to_pass": ["tests/test_parser.py::test_empty_window"],
            "duration_s": 10.0 + i,
        },
        messages=messages,
        harness="opencode",
        model_id="Qwen/Qwen3-Coder-Next",
        started_at="2026-10-21T10:00:00Z",
        generator={
            "source_id": "mentee_synth_v1",
            "license": "Apache-2.0",
            "license_verified": True,
            "trainable": True,
            "teacher": {
                "model": "Qwen/Qwen3-Coder-Next",
                "revision": "main",
                "license": "Apache-2.0",
                "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
            },
        },
    )


def test_collector_loop_appends_valid_rows(tmp_path: Path):
    out = tmp_path / "train.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    tier_a = TierAValidator(validator=VALIDATOR)
    schedule = IntegritySchedule(first_check_at=10, seed=1)
    reporter = Reporter()
    sidecar = Sidecar(out).load()

    with JsonlAppender(out, fsync_every=5) as app, JsonlAppender(rejected) as rej:
        sidecar.open()
        for i in range(25):
            record, problems = build_record(synthetic_trajectory(i), f"traj_2026-10-21_{i:06d}")
            assert record is not None, problems
            verdict = tier_a.validate(record)
            if not verdict.ok:
                rej.append({"id": record["id"], "reasons": verdict.reasons})
                reporter.reject(verdict.reasons)
                continue
            app.append(record)
            sidecar.add_id(record["id"])
            reporter.accept(record)
            if schedule.due(app.lines):
                report = verify(
                    out, believed_lines=app.lines, believed_bytes=app.bytes, deep=schedule.is_deep()
                )
                assert report.ok, report.summary()
                schedule.mark_checked()
                app.checkpoint()
                sidecar.checkpoint(app.lines, app.bytes, record["id"], dict(reporter.rejections))
        sidecar.close()

    assert reporter.accepted == 25
    assert reporter.rejected == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 25
    for line in lines:
        obj = json.loads(line)
        assert obj["id"].startswith("traj_")
    # every record validates against the canonical schema
    for line in lines:
        assert not list(VALIDATOR.iter_errors(json.loads(line)))


def test_collector_loop_rejects_bad_rows_and_keeps_good_ones(tmp_path: Path):
    out = tmp_path / "train.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    tier_a = TierAValidator(validator=VALIDATOR)
    reporter = Reporter()

    with JsonlAppender(out) as app, JsonlAppender(rejected) as rej:
        for i in range(10):
            raw = synthetic_trajectory(i)
            if i % 3 == 0:
                # a truncated tool result — the most destructive silent poison
                raw.messages[3].tool_results[0]["truncated"] = True
            record, _ = build_record(raw, f"traj_2026-10-21_{i:06d}")
            verdict = tier_a.validate(record)
            if not verdict.ok:
                rej.append({"id": record["id"], "reasons": verdict.reasons})
                reporter.reject(verdict.reasons)
                continue
            app.append(record)
            reporter.accept(record)

    # i in {0,3,6,9} carry a truncated tool result -> 4 rejected, 6 accepted
    assert reporter.accepted == 6
    assert reporter.rejected == 4
    assert reporter.rejections["no_truncated_output"] == 4
    assert len(out.read_text(encoding="utf-8").splitlines()) == 6
    assert len(rejected.read_text(encoding="utf-8").splitlines()) == 4


def test_duplicate_ids_are_detected_by_the_sidecar(tmp_path: Path):
    out = tmp_path / "train.jsonl"
    sidecar = Sidecar(out).load()
    sidecar.open()
    assert not sidecar.has_id("traj_2026-10-21_000001")
    sidecar.add_id("traj_2026-10-21_000001")
    assert sidecar.has_id("traj_2026-10-21_000001")
    sidecar.close()

    reloaded = Sidecar(out).load()
    assert reloaded.has_id("traj_2026-10-21_000001")


def test_resume_after_a_crash_keeps_the_file_consistent(tmp_path: Path):
    out = tmp_path / "train.jsonl"
    tier_a = TierAValidator(validator=VALIDATOR)

    # first session
    with JsonlAppender(out) as app:
        for i in range(8):
            record, _ = build_record(synthetic_trajectory(i), f"traj_2026-10-21_{i:06d}")
            assert tier_a.validate(record).ok
            app.append(record)
        app.close()

    # simulate a crash: a half-written record at the end
    with open(out, "ab") as fh:
        fh.write(b'{"schema_version":"1.0.0","id":"traj_2026-10')

    # second session resumes
    with JsonlAppender(out) as app:
        assert app.healed_bytes > 0
        assert app.lines == 8
        for i in range(8, 12):
            record, _ = build_record(synthetic_trajectory(i), f"traj_2026-10-21_{i:06d}")
            assert tier_a.validate(record).ok
            app.append(record)
        final_lines = app.lines

    report = verify(out, believed_lines=final_lines, deep=True, schema=SCHEMA, validator=VALIDATOR)
    assert report.ok, report.summary()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 12
