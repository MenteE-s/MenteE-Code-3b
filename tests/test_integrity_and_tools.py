"""Tests for the integrity schedule and the tool-schema shuffler."""

from __future__ import annotations

from src.collect.integrity import IntegritySchedule
from src.collect.schema_tools import ToolShuffler


# --------------------------------------------------------------------- schedule
def test_first_check_is_at_the_configured_row():
    schedule = IntegritySchedule(first_check_at=10, seed=1)
    assert schedule.next_threshold() == 10
    assert schedule.due(9) is False
    assert schedule.due(10) is True


def test_schedule_widens():
    schedule = IntegritySchedule(first_check_at=10, gaps=[30, 45, 60, 90], seed=7)
    plan = schedule.plan()
    assert plan[0] == 10
    # gaps must grow, not shrink
    deltas = [plan[i + 1] - plan[i] for i in range(len(plan) - 1)]
    assert deltas[0] >= 20
    assert deltas[-1] >= 100


def test_schedule_jitter_varies_the_plan():
    a = IntegritySchedule(seed=1).plan()[:6]
    b = IntegritySchedule(seed=2).plan()[:6]
    assert a != b


def test_schedule_marks_every_nth_check_deep():
    schedule = IntegritySchedule(full_rescan_every=5)
    deeps = []
    for _ in range(10):
        deeps.append(schedule.is_deep())
        schedule.mark_checked()
    assert deeps.count(True) == 2


def test_schedule_advances_after_a_check():
    schedule = IntegritySchedule(first_check_at=10, seed=3)
    first = schedule.next_threshold()
    schedule.mark_checked()
    assert schedule.next_threshold() > first


# --------------------------------------------------------------------- shuffler
TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply a patch.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}},
            "required": ["path", "old_string"],
        },
    },
    {
        "name": "execute_command",
        "description": "Run a command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def test_shuffle_is_deterministic_for_a_seed():
    a = ToolShuffler().shuffle(TOOLS, seed=42)
    b = ToolShuffler().shuffle(TOOLS, seed=42)
    assert [t["name"] for t in a.tools] == [t["name"] for t in b.tools]
    assert a.name_mapping == b.name_mapping


def test_different_seeds_produce_different_variants():
    variants = {tuple(t["name"] for t in ToolShuffler().shuffle(TOOLS, seed=s).tools) for s in range(30)}
    assert len(variants) > 1


def test_shuffle_never_creates_duplicate_names():
    for seed in range(200):
        result = ToolShuffler().shuffle(TOOLS, seed=seed)
        names = [t["name"] for t in result.tools]
        assert len(names) == len(set(names)), f"collision at seed {seed}: {names}"


def test_shuffle_never_empties_the_tool_set():
    for seed in range(200):
        result = ToolShuffler().shuffle(TOOLS, seed=seed)
        assert result.tools


def test_shuffle_keeps_required_params_in_sync():
    """Renaming a parameter must rename it in `required` too."""
    for seed in range(100):
        result = ToolShuffler().shuffle(TOOLS, seed=seed)
        for tool in result.tools:
            required = tool["parameters"].get("required", [])
            properties = set(tool["parameters"].get("properties", {}))
            for r in required:
                assert r in properties, f"{tool['name']}: required {r} not in properties"


def test_shuffle_preserves_tool_descriptions_and_types():
    for seed in range(50):
        result = ToolShuffler().shuffle(TOOLS, seed=seed)
        for tool in result.tools:
            assert tool.get("description")
            assert tool["parameters"].get("type") == "object"


def test_variant_for_is_task_keyed():
    a = ToolShuffler().variant_for(TOOLS, "task-0001")
    b = ToolShuffler().variant_for(TOOLS, "task-0001")
    c = ToolShuffler().variant_for(TOOLS, "task-0002")
    assert a.name_mapping == b.name_mapping
    # different tasks may differ, but the mapping is at least well-formed
    assert isinstance(c.name_mapping, dict)


def test_reverse_mapping_round_trips():
    result = ToolShuffler().shuffle(TOOLS, seed=11)
    for original, shuffled in result.name_mapping.items():
        assert result.reverse()[shuffled] == original
