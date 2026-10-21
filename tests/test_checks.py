"""Tests for the Tier A checks and the record builder.

Every check needs a passing fixture and a failing fixture — a check that has
never been seen to fail is a check you cannot trust to fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collect.checks import (
    check_arguments_raw_agreement,
    check_honest_conclusion,
    check_length_bounds,
    check_no_hallucinated_paths,
    check_no_secrets,
    check_no_tool_loop,
    check_no_truncated_output,
    check_provenance,
    check_tool_call_pairing,
    check_tool_names,
    check_turn_sequence,
    check_verification_consistency,
    run_tier_a,
)
from src.collect.normalizer import CommonMessage, RawTrajectory, build_record
from src.collect.validate import load_schema, make_validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
VALIDATOR = make_validator(SCHEMA)


# --------------------------------------------------------------------- fixtures
def make_record(**overrides) -> dict:
    record = {
        "schema_version": "1.0.0",
        "id": "traj_2026-10-21_000001",
        "provenance": {
            "source_id": "mentee_synth_v1",
            "source_url": "internal",
            "source_sha256": "abc",
            "license": "Apache-2.0",
            "license_verified": True,
            "redistributable": True,
            "trainable": True,
            "teacher": {
                "model": "Qwen/Qwen3-Coder-Next",
                "revision": "main",
                "license": "Apache-2.0",
                "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
            },
            "generator": {"code_version": "git:abc", "prompt_template": "t", "harness": "opencode"},
            "collected_at": "2026-10-21T10:00:00Z",
            "collected_by": "test",
        },
        "task": {
            "instruction": "Fix the off-by-one in parse_window so an empty window returns [].",
            "language": "python",
            "task_type": "bug_fix",
            "difficulty": "medium",
            "constraints": [],
            "acceptance_criteria": [],
            "outcome": "solved",
        },
        "environment": {
            "repo": "acme/windowlib",
            "repo_license": "MIT",
            "base_commit": "9f2c1ab",
            "setup_script": "pip install -e .",
            "test_command": "pytest -q tests/",
            "fail_to_pass": ["tests/test_parser.py::test_empty_window"],
            "pass_to_pass": ["tests/test_parser.py::test_basic"],
        },
        "trajectory": {
            "turns": [
                {"index": 0, "role": "system", "content": "You are a coding agent."},
                {"index": 1, "role": "user", "content": "Fix the off-by-one."},
                {
                    "index": 2,
                    "role": "assistant",
                    "content": "Reading the parser first.",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "name": "read_file",
                            "arguments": {"path": "src/parser.py"},
                            "arguments_raw": '{"path":"src/parser.py"}',
                        }
                    ],
                },
                {
                    "index": 3,
                    "role": "tool_result",
                    "content": (
                        "def parse_window(tokens, size):\n"
                        "    if not tokens:\n"
                        "        return tokens.index('[')\n"
                        "    start = tokens.index('[')\n"
                        "    end = tokens.index(']')\n"
                        "    body = tokens[start + 1:end]\n"
                        "    return body[0:size - 1]\n"
                    ),
                    "tool_call_id": "call_0",
                    "name": "read_file",
                    "truncated": False,
                    "content_bytes": 60,
                },
                {
                    "index": 4,
                    "role": "assistant",
                    "content": "The slice drops the last element. Patching the guard.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "apply_patch",
                            "arguments": {"path": "src/parser.py"},
                            "arguments_raw": '{"path":"src/parser.py"}',
                        }
                    ],
                },
                {
                    "index": 5,
                    "role": "tool_result",
                    "content": "Applied patch (+1 -1)",
                    "tool_call_id": "call_1",
                    "name": "apply_patch",
                    "truncated": False,
                    "content_bytes": 24,
                },
                {
                    "index": 6,
                    "role": "assistant",
                    "content": (
                        "The guard on line 42 returns tokens.index('[') instead of an empty list, "
                        "which is what raises IndexError when the token list is empty. I replaced "
                        "the guard body with an empty list and left the public signature and the "
                        "slice behaviour untouched. All tests pass now: 4 passed in 0.31s."
                    ),
                },
            ],
            "patches": [
                {"path": "src/parser.py", "diff": "--- a\n+++ b\n", "applied": True}
            ],
            "n_turns": 7,
            "stop_reason": "completed",
            "duration_s": 42.0,
            "total_tokens": {"input": 1200, "output": 180},
        },
        "verification": {
            "status": "pass",
            "verifier": "docker-runner@1.2.0",
            "verified_at": "2026-10-21T10:05:00Z",
            "tests": {
                "fail_to_pass": {"before": "FAIL", "after": "PASS"},
                "pass_to_pass": {"before": "PASS", "after": "PASS", "total": 3, "regressed": 0},
            },
            "regressions": [],
        },
        "quality": {
            "verified": True,
            "score": 0.9,
            "checks": {
                "schema_valid": True,
                "no_truncated_tool_output": True,
                "has_failure_path": True,
                "no_secrets": True,
                "no_hallucinated_paths": True,
                "patch_applies": True,
                "regressions": 0,
                "token_length_ok": True,
                "dedup_cluster": None,
            },
            "rejected_by": [],
            "filter_notes": [],
        },
        "split": {
            "partition": "train",
            "reason": "commit_date_before_2026-07-01",
            "split_group": "acme/windowlib@9f2c1ab",
        },
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------- happy path
def test_fixture_validates_against_the_schema():
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(SCHEMA).iter_errors(make_record()))
    assert not errors, [e.message for e in errors]


def test_all_tier_a_checks_pass_on_the_fixture():
    results = run_tier_a(make_record(), validator=VALIDATOR, known_tools=["read_file", "apply_patch"])
    failed = [f"{r.name}: {r.reason}" for r in results if not r.ok]
    assert not failed, failed


# --------------------------------------------------------------------- individual checks
def test_turn_sequence_detects_bad_index():
    record = make_record()
    record["trajectory"]["turns"][2]["index"] = 99
    assert not check_turn_sequence(record).ok


def test_turn_sequence_detects_count_mismatch():
    record = make_record()
    record["trajectory"]["n_turns"] = 99
    assert not check_turn_sequence(record).ok


def test_tool_call_pairing_detects_missing_result():
    record = make_record()
    record["trajectory"]["turns"] = [
        t for t in record["trajectory"]["turns"] if t.get("tool_call_id") != "call_0"
    ]
    for i, t in enumerate(record["trajectory"]["turns"]):
        t["index"] = i
    record["trajectory"]["n_turns"] = len(record["trajectory"]["turns"])
    assert not check_tool_call_pairing(record).ok


def test_tool_call_pairing_detects_orphan_result():
    record = make_record()
    record["trajectory"]["turns"][3]["tool_call_id"] = "call_missing"
    assert not check_tool_call_pairing(record).ok


def test_arguments_raw_must_agree_with_arguments():
    record = make_record()
    record["trajectory"]["turns"][2]["tool_calls"][0]["arguments_raw"] = '{"path":"other.py"}'
    assert not check_arguments_raw_agreement(record).ok


def test_arguments_raw_must_be_present():
    record = make_record()
    del record["trajectory"]["turns"][2]["tool_calls"][0]["arguments_raw"]
    assert not check_arguments_raw_agreement(record).ok


def test_tool_names_must_be_advertised():
    record = make_record()
    assert check_tool_names(record, known_tools=["read_file", "apply_patch"]).ok
    assert not check_tool_names(record, known_tools=["read_file"]).ok


def test_truncated_tool_output_is_rejected():
    record = make_record()
    record["trajectory"]["turns"][3]["truncated"] = True
    assert not check_no_truncated_output(record).ok


def test_secret_detection():
    record = make_record()
    record["trajectory"]["turns"][1]["content"] = "here is the key sk-abcdefghijklmnopqrstuvwxyz123456"
    assert not check_no_secrets(record).ok


def test_placeholder_is_not_a_secret():
    record = make_record()
    record["trajectory"]["turns"][1]["content"] = "set OPENAI_API_KEY=your-api-key-here"
    assert check_no_secrets(record).ok


def test_tool_loop_detection():
    record = make_record()
    call = {
        "id": "loop",
        "name": "read_file",
        "arguments": {"path": "a.py"},
        "arguments_raw": '{"path":"a.py"}',
    }
    turns = record["trajectory"]["turns"]
    for i in range(3):
        turns.append({"index": len(turns), "role": "assistant", "content": "", "tool_calls": [dict(call, id=f"loop{i}")]})
    record["trajectory"]["n_turns"] = len(turns)
    assert not check_no_tool_loop(record, max_identical=3).ok
    assert check_no_tool_loop(record, max_identical=5).ok


def test_length_bounds():
    record = make_record()
    assert check_length_bounds(record, min_chars=1).ok
    assert not check_length_bounds(record, min_chars=100000).ok


def test_provenance_gate():
    record = make_record()
    assert check_provenance(record).ok
    record["provenance"]["trainable"] = False
    assert not check_provenance(record).ok
    record["provenance"]["trainable"] = True
    record["provenance"]["license_verified"] = False
    assert not check_provenance(record).ok


def test_verification_consistency():
    record = make_record()
    assert check_verification_consistency(record).ok
    record["verification"]["status"] = "pass"
    record["environment"]["fail_to_pass"] = []
    assert not check_verification_consistency(record).ok


def test_verification_skipped_cannot_claim_verified():
    record = make_record()
    record["verification"]["status"] = "skipped"
    record["quality"]["verified"] = True
    assert not check_verification_consistency(record).ok


def test_hallucinated_paths():
    record = make_record()
    assert check_no_hallucinated_paths(record, {"src/parser.py"}).ok
    assert not check_no_hallucinated_paths(record, {"other.py"}).ok


def test_honest_conclusion_rejects_false_success_claim():
    record = make_record()
    record["verification"]["status"] = "fail"
    record["trajectory"]["turns"][-1]["content"] = "All tests pass now, the bug is fixed."
    assert not check_honest_conclusion(record).ok


def test_failed_but_honest_is_accepted():
    record = make_record()
    record["verification"]["status"] = "fail"
    record["task"]["outcome"] = "failed_but_honest"
    record["trajectory"]["turns"][-1]["content"] = (
        "I could not reproduce the failure in this environment. The change is applied "
        "but the test still fails locally, so this is not resolved."
    )
    assert check_honest_conclusion(record).ok


# --------------------------------------------------------------------- builder
def _raw_trajectory() -> RawTrajectory:
    messages = [
        CommonMessage(role="system", content="You are a coding agent."),
        CommonMessage(role="user", content="Fix the off-by-one."),
        CommonMessage(
            role="assistant",
            content="Reading the parser.",
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
            content="def parse_window(...): ...",
            tool_results=[
                {
                    "tool_call_id": "call_0",
                    "name": "read_file",
                    "content": "def parse_window(...): ...",
                    "truncated": False,
                }
            ],
        ),
        CommonMessage(role="assistant", content="Patched the guard."),
    ]
    return RawTrajectory(
        task={
            "instruction": "Fix the off-by-one in parse_window.",
            "repo": "acme/windowlib",
            "base_commit": "9f2c1ab",
            "language": "python",
            "task_type": "bug_fix",
            "fail_to_pass": ["tests/test_parser.py::test_empty_window"],
            "duration_s": 12.0,
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


def test_build_record_produces_a_valid_record():
    record, problems = build_record(_raw_trajectory(), "traj_2026-10-21_000007")
    assert record is not None, problems
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(SCHEMA).iter_errors(record))
    assert not errors, [e.message for e in errors]
    assert record["id"] == "traj_2026-10-21_000007"
    assert record["trajectory"]["n_turns"] == len(record["trajectory"]["turns"])


def test_build_record_pairs_tool_results():
    record, _ = build_record(_raw_trajectory(), "traj_2026-10-21_000008")
    results = [t for t in record["trajectory"]["turns"] if t["role"] == "tool_result"]
    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call_0"


def test_build_record_keeps_unverified_rows_flagged():
    record, _ = build_record(_raw_trajectory(), "traj_2026-10-21_000009")
    assert record["verification"]["status"] == "skipped"
    assert record["quality"]["verified"] is False
