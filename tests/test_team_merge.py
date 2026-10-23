"""Tests for multi-collector operation: namespacing, partitioning and merging.

The scenario: four collectors run on four machines, each pushes their own
folder, and one person merges. Two failure modes must be impossible:

1. two collectors producing the same trajectory id (silently drops real data);
2. one misconfigured collector injecting garbage that nobody notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.collect.cli import _record_id, _slug
from src.collect.merge import content_key, merge_datasets
from src.collect.state import Sidecar
from src.collect.validate import load_schema, make_validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(REPO / "schemas" / "mentee-trajectory-v1.schema.json")
VALIDATOR = make_validator(SCHEMA)


def _record(rid: str, instruction: str, collector: str, verified: bool = True) -> dict:
    return {
        "schema_version": "1.0.0",
        "id": rid,
        "provenance": {
            "source_id": "mentee_synth_v1",
            "source_url": "internal",
            "source_sha256": "abc",
            "license": "Apache-2.0",
            "license_verified": True,
            "redistributable": True,
            "trainable": True,
            "teacher": {
                "model": "cohere/north-mini-code",
                "revision": "free",
                "license": "Apache-2.0",
                "sampling": {"temperature": 0.7},
            },
            "generator": {"harness": "opencode", "code_version": "git:abc", "prompt_template": "t"},
            "collected_at": "2026-10-23T10:00:00Z",
            "collected_by": collector,
        },
        "task": {
            "instruction": instruction,
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
            "setup_script": "",
            "test_command": "pytest -q",
            "fail_to_pass": ["tests/test_parser.py::test_empty_window"],
            "pass_to_pass": [],
        },
        "trajectory": {
            "turns": [
                {"index": 0, "role": "system", "content": "You are a coding agent."},
                {"index": 1, "role": "user", "content": instruction},
                {
                    "index": 2,
                    "role": "assistant",
                    "content": "Reading the parser.",
                    "tool_calls": [
                        {
                            "id": "c0",
                            "name": "read_file",
                            "arguments": {"path": "src/parser.py"},
                            "arguments_raw": '{"path": "src/parser.py"}',
                        }
                    ],
                },
                {
                    "index": 3,
                    "role": "tool_result",
                    "content": "def parse_window(tokens, size):\n    return tokens[0:size-1]\n",
                    "tool_call_id": "c0",
                    "name": "read_file",
                    "truncated": False,
                    "content_bytes": 60,
                },
                {"index": 4, "role": "assistant", "content": "The guard returns the wrong value when the list is empty, so I replaced it with an empty list and all tests pass."},
            ],
            "patches": [{"path": "src/parser.py", "diff": "--- a\n+++ b\n", "applied": True}],
            "n_turns": 5,
            "stop_reason": "completed",
            "duration_s": 12.0,
            "total_tokens": {"input": 900, "output": 120},
        },
        "verification": {
            "status": "pass" if verified else "skipped",
            "verifier": "docker-runner@1.2.0",
            "verified_at": "2026-10-23T10:05:00Z",
            "tests": {
                "fail_to_pass": {"before": "FAIL", "after": "PASS"},
                "pass_to_pass": {"before": "PASS", "after": "PASS", "total": 2, "regressed": 0},
            },
            "regressions": [],
        },
        "quality": {
            "verified": verified,
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


# --------------------------------------------------------------------- ids
def test_collector_slug_prevents_id_collisions():
    """Two collectors starting the same day must not produce the same id."""
    a = Sidecar(Path("unused")).load()
    b = Sidecar(Path("unused")).load()
    id_a = _record_id(a, "syab")
    id_b = _record_id(b, "ahmed")
    assert id_a != id_b
    assert id_a.startswith("traj_syab_")
    assert id_b.startswith("traj_ahmed_")


def test_collector_slug_is_normalised():
    assert _slug("Syed Syab") == _slug("syed-syab")
    assert _slug("Ahmed!") == "ahmed"


def test_id_without_collector_still_matches_schema():
    from jsonschema import Draft202012Validator

    pattern = "^" + SCHEMA["properties"]["id"]["pattern"] + "$"
    validator = Draft202012Validator({"type": "string", "pattern": pattern})
    assert validator.is_valid(_record_id(Sidecar(Path("unused")).load(), ""))
    assert validator.is_valid("traj_syab_2026-10-23_000001")
    assert not validator.is_valid("traj_-2026-10-23_000001")


# --------------------------------------------------------------------- merge
def _write_collector(tmp_path: Path, name: str, records: list[dict]) -> Path:
    p = tmp_path / name / "train.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_merge_combines_two_collectors(tmp_path: Path):
    a = _write_collector(
        tmp_path, "syab",
        [_record(f"traj_syab_2026-10-23_{i:06d}", f"Fix bug {i}", "syab") for i in range(5)],
    )
    b = _write_collector(
        tmp_path, "ahmed",
        [_record(f"traj_ahmed_2026-10-23_{i:06d}", f"Fix other bug {i}", "ahmed") for i in range(4)],
    )
    out = tmp_path / "merged" / "v0.1.jsonl"
    report = merge_datasets([a, b], out, validator=VALIDATOR)

    assert report.ok
    assert report.total_accepted == 9
    assert report.total_duplicate_id == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 9
    ids = [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(set(ids))


def test_merge_drops_same_content_from_two_collectors(tmp_path: Path):
    """Same task, same attempt, two collectors -> one row, not two."""
    same = _record("traj_syab_2026-10-23_000001", "Fix bug 1", "syab")
    twin = _record("traj_ahmed_2026-10-23_000001", "Fix bug 1", "ahmed")  # identical attempt
    a = _write_collector(tmp_path, "s1", [same])
    b = _write_collector(tmp_path, "s2", [twin])
    out = tmp_path / "merged.jsonl"
    report = merge_datasets([a, b], out, validator=VALIDATOR)
    assert report.total_accepted == 1
    assert report.total_duplicate_content == 1


def test_merge_flags_a_bad_collector(tmp_path: Path):
    """One collector with a corrupt file must be visible, not silently merged."""
    good = _write_collector(
        tmp_path, "syab",
        [_record(f"traj_syab_2026-10-23_{i:06d}", f"Fix bug {i}", "syab") for i in range(3)],
    )
    bad = tmp_path / "bilal" / "train.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"id": "traj_bilal_2026-10-23_000001"\n', encoding="utf-8")  # partial line

    out = tmp_path / "merged" / "v0.1.jsonl"
    report = merge_datasets([good, bad], out, validator=VALIDATOR, require_integrity=False)
    bad_stat = next(s for s in report.sources if "bilal" in s.path)
    assert bad_stat.integrity_ok is False
    assert "bilal" in str(bad_stat.integrity_note)


def test_merge_rejects_schema_invalid_rows(tmp_path: Path):
    good = _record("traj_syab_2026-10-23_000001", "Fix bug 1", "syab")
    bad = _record("traj_syab_2026-10-23_000002", "Fix bug 2", "syab")
    bad["verification"]["status"] = "nonsense"  # enum violation
    p = _write_collector(tmp_path, "syab", [good, bad])
    out = tmp_path / "merged" / "v0.1.jsonl"
    report = merge_datasets([p], out, validator=VALIDATOR)
    assert report.total_accepted == 1
    assert report.total_schema_invalid == 1


def test_content_key_ignores_collector_identity():
    a = _record("traj_syab_2026-10-23_000001", "Fix bug 1", "syab")
    b = _record("traj_ahmed_2026-10-23_000001", "Fix bug 1", "ahmed")
    assert content_key(a) == content_key(b)


def test_content_key_distinguishes_different_attempts():
    a = _record("traj_syab_2026-10-23_000001", "Fix bug 1", "syab")
    b = _record("traj_syab_2026-10-23_000002", "Fix bug 1", "syab")
    b["trajectory"]["turns"][2]["content"] = "A different first step: searching instead of reading."
    assert content_key(a) != content_key(b)
