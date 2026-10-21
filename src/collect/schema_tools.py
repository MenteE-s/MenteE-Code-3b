"""Tool-schema shuffling: teach the model to *read* the schema, not memorise a tool list.

The problem
-----------
If every training trajectory uses the same tool names and parameter names, the model
learns a lookup table: "when the user asks for a file, call ``read_file``". That
model looks excellent in the harness you trained on and falls apart in any other.

The fix
-------
The tool schema lives in the **request**, not in the weights. So we can vary it for
free: rename the tools, rename the parameters, drop tools, add tools. The model then
has to read the schema it was given in order to act, which is the behaviour that
generalises across harnesses (Claude Code, Cline, Kilo, Roo, OpenCode, our own).

This module is deliberately *not* tied to any one harness. It operates on a plain
list of OpenAI-style tool schemas.

Determinism
-----------
Shuffles are seeded, so a run is reproducible: the same task id always produces the
same schema variant.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolShuffler", "ShuffleResult", "DEFAULT_NAME_POOLS", "DEFAULT_PARAM_POOLS"]


# Tool name variants. The first entry of each tuple is the original name; the rest
# are alternatives. Names are grouped by *function*, not by vendor, so the model
# learns the mapping from a described function to a call.
DEFAULT_NAME_POOLS: dict[str, tuple[str, ...]] = {
    "read_file": ("read", "view_file", "open_file", "read_source", "cat"),
    "write_file": ("write", "save_file", "create_file", "put_file", "write_content"),
    "apply_patch": ("str_replace_editor", "patch_file", "edit_file", "apply_edit", "replace_in_file"),
    "execute_command": ("run_command", "shell", "bash", "exec", "run_shell", "terminal"),
    "list_files": ("ls", "list_dir", "tree", "glob", "dir"),
    "search_code": ("grep", "search", "find", "ripgrep", "code_search"),
    "search_files": ("find_files", "glob_files", "file_search", "locate"),
    "git_status": ("status", "git_status", "vcs_status"),
    "git_diff": ("diff", "git_diff", "show_diff"),
    "run_tests": ("test", "run_test", "pytest", "run_tests"),
    "get_file_info": ("stat", "file_info", "file_stat", "info"),
    "move_path": ("move", "rename", "mv", "move_file"),
}

# Parameter name variants, keyed by the *original* parameter name.
DEFAULT_PARAM_POOLS: dict[str, tuple[str, ...]] = {
    "path": ("path", "file_path", "filename", "file", "target"),
    "command": ("command", "cmd", "script", "shell_command"),
    "content": ("content", "text", "body", "new_content"),
    "old_string": ("old_string", "old", "from", "search", "old_str"),
    "new_string": ("new_string", "new", "to", "replace", "new_str"),
    "start_line": ("start_line", "from_line", "offset", "begin", "line_start"),
    "end_line": ("end_line", "to_line", "limit", "end", "line_end"),
    "pattern": ("pattern", "query", "regex", "search"),
    "recursive": ("recursive", "deep", "r"),
    "description": ("description", "summary", "desc"),
}

# Extra tools that can be *added* to a tool set. They exist to make the advertised
# set vary in size and shape, not to teach anything specific.
_EXTRA_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_directory",
        "description": "List the files and directories under a path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory to list."}},
            "required": ["path"],
        },
    },
    {
        "name": "read_many_files",
        "description": "Read several files at once and return their contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["paths"],
        },
    },
    {
        "name": "search_replace",
        "description": "Replace an exact string in a file with a new string.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "get_problem_context",
        "description": "Describe the current problem, the files involved, and the relevant source code.",
        "parameters": {
            "type": "object",
            "properties": {"type": {"type": "string", "description": "Type of problem to solve."}},
            "required": ["type"],
        },
    },
    {
        "name": "attempt_completion",
        "description": "Present the result of the task for review.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


@dataclass
class ShuffleResult:
    """The outcome of shuffling one tool set."""

    tools: list[dict[str, Any]]
    name_mapping: dict[str, str] = field(default_factory=dict)  # original -> shuffled
    param_mapping: dict[str, dict[str, str]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)

    def reverse(self) -> dict[str, str]:
        """shuffled name -> original name"""
        return {v: k for k, v in self.name_mapping.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_mapping": self.name_mapping,
            "param_mapping": self.param_mapping,
            "dropped": self.dropped,
            "added": self.added,
        }


class ToolShuffler:
    """Produce reproducible variants of a tool set."""

    def __init__(
        self,
        name_pools: dict[str, tuple[str, ...]] | None = None,
        param_pools: dict[str, tuple[str, ...]] | None = None,
        rename_probability: float = 0.8,
        rename_params_probability: float = 0.7,
        drop_probability: float = 0.25,
        add_probability: float = 0.35,
        max_dropped: int = 3,
    ):
        self.name_pools = name_pools or DEFAULT_NAME_POOLS
        self.param_pools = param_pools or DEFAULT_PARAM_POOLS
        self.rename_probability = rename_probability
        self.rename_params_probability = rename_params_probability
        self.drop_probability = drop_probability
        self.add_probability = add_probability
        self.max_dropped = max_dropped

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def seed_for(task_id: str, salt: str = "") -> int:
        """Stable seed derived from a task id, so variants are reproducible."""
        digest = hashlib.sha256(f"{task_id}:{salt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def _rename_tool(self, name: str, rng: random.Random) -> str:
        pool = self.name_pools.get(name)
        if not pool or len(pool) == 1:
            return name
        if rng.random() > self.rename_probability:
            return name
        alternatives = [n for n in pool if n != name]
        return rng.choice(alternatives) if alternatives else name

    def _rename_params(self, tool: dict, rng: random.Random) -> tuple[dict, dict[str, str]]:
        props = tool.get("parameters", {}).get("properties")
        if not isinstance(props, dict) or not props:
            return tool, {}
        if rng.random() > self.rename_params_probability:
            return tool, {}
        mapping: dict[str, str] = {}
        new_props: dict[str, Any] = {}
        for key, value in props.items():
            pool = self.param_pools.get(key)
            new_key = key
            if pool and len(pool) > 1 and rng.random() <= self.rename_params_probability:
                options = [p for p in pool if p != key]
                if options:
                    new_key = rng.choice(options)
            mapping[key] = new_key
            new_props[new_key] = value
        required = tool.get("parameters", {}).get("required")
        if isinstance(required, list):
            tool["parameters"]["required"] = [mapping.get(r, r) for r in required]
        tool["parameters"]["properties"] = new_props
        return tool, mapping

    # ------------------------------------------------------------------ public
    def shuffle(self, tools: list[dict[str, Any]], seed: int) -> ShuffleResult:
        """Produce a variant of ``tools``."""
        rng = random.Random(seed)
        result = ShuffleResult(tools=[])

        kept: list[dict[str, Any]] = []
        for tool in tools:
            name = tool.get("name", "")
            if name and rng.random() < self.drop_probability:
                result.dropped.append(name)
                if len(result.dropped) >= self.max_dropped:
                    continue
            else:
                kept.append(tool)

        # Never hand the model an empty tool set.
        if not kept:
            kept = list(tools)

        used_names: set[str] = {t.get("name") for t in kept}
        for tool in kept:
            tool = _deepcopy(tool)
            original = tool.get("name", "")
            new_name = self._rename_tool(original, rng)
            # Avoid collisions with a name already in the set.
            if new_name in used_names and new_name != original:
                new_name = original
            used_names.add(new_name)
            if new_name != original:
                result.name_mapping[original] = new_name
            tool["name"] = new_name
            tool, pmap = self._rename_params(tool, rng)
            if pmap:
                result.param_mapping[new_name] = pmap
            result.tools.append(tool)

        if rng.random() < self.add_probability:
            extra = _deepcopy(rng.choice(_EXTRA_TOOLS))
            if extra.get("name") not in used_names:
                result.tools.append(extra)
                result.added.append(extra["name"])
                used_names.add(extra["name"])

        return result

    def variant_for(self, tools: list[dict[str, Any]], task_id: str, salt: str = "") -> ShuffleResult:
        """Deterministic shuffle keyed by task id."""
        return self.shuffle(tools, self.seed_for(task_id, salt))


def _deepcopy(obj: Any) -> Any:
    import copy

    return copy.deepcopy(obj)
