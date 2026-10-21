"""Configuration loading for the collector.

The config is the single source of truth for a run. It is YAML (or JSON), it is
merged over the defaults below, and the resolved config is written into the
run's report so a run can be reproduced from its own artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["DEFAULTS", "load_config", "save_config"]

DEFAULTS: dict[str, Any] = {
    "source": {
        "kind": "harness",  # harness | replay
        "harness": "opencode",
        "tasks_path": "data/tasks/tasks.jsonl",
        "opencode": {
            "server_url": "http://127.0.0.1:4096",
            "auth_user": "opencode",
            "auth_password_env": "OPENCODE_SERVER_PASSWORD",
            "model": "",
            "agent": "",
            "timeout_s": 1800,
        },
        "replay": {"path": ""},
        "teacher": {
            "model": "Qwen/Qwen3-Coder-Next",
            "revision": "main",
            "license": "Apache-2.0",
            "base_url": "http://127.0.0.1:8000/v1",
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40},
        },
        "proxy": {
            "enabled": False,
            "listen_host": "127.0.0.1",
            "listen_port": 8010,
            "log_dir": "data/raw/proxy_logs",
        },
        "generator": {
            "source_id": "mentee_synth_v1",
            "source_url": "internal",
            "source_sha256": "",
            "license": "Apache-2.0",
            "license_verified": True,
            "redistributable": True,
            "trainable": True,
            "collected_by": "collector",
            "code_version": "",
            "prompt_template": "",
        },
        "tool_schema_shuffle": {
            "enabled": True,
            "probability": 0.35,
            "rename_probability": 0.8,
            "rename_params_probability": 0.7,
            "drop_probability": 0.25,
            "add_probability": 0.35,
        },
    },
    "output": {
        "path": "data/processed/v1.0.0/train.jsonl",
        "rejected_path": "data/processed/v1.0.0/rejected.jsonl",
        "fsync_every": 10,
        "resume": True,
    },
    "validation": {
        "schema_path": "schemas/mentee-trajectory-v1.schema.json",
        "tier_b_sample_rate": 0.02,
        "tier_b_timeout_s": 900,
        "require_verified": False,
        "min_chars": 400,
        "max_chars": 400000,
        "max_identical_tool_calls": 3,
    },
    "integrity": {
        "enabled": True,
        "first_check_at": 10,
        "gaps": [30, 45, 60, 90],
        "steady_every": 200,
        "jitter_ratio": 0.30,
        "full_rescan_every": 5,
        "on_failure": "abort",
    },
    "budget": {
        "max_rows": None,
        "max_minutes": None,
        "max_generation_tokens": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required for YAML configs. Install it, or use a .json config."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config file merged over the defaults."""
    if path is None:
        return json.loads(json.dumps(DEFAULTS))
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    return _deep_merge(DEFAULTS, _read(p))


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
