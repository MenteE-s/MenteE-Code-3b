"""MenteE Code 3B — data collection toolkit.

A single loop: **generate one trajectory, validate it, append it, repeat.**

The dataset file is only ever appended to. The appender's guarantees (never
truncate, never leave a partial line, never silently write nothing) and the
widening integrity schedule are what make a long unattended run safe.
"""

from .appender import AppendError, JsonlAppender
from .budget import Budget
from .config import DEFAULTS, load_config, save_config
from .integrity import IntegrityReport, IntegritySchedule, verify
from .normalizer import (
    CommonMessage,
    GenericAdapter,
    OpenCodeAdapter,
    RawTrajectory,
    build_record,
)
from .report import Reporter
from .schema_tools import ToolShuffler
from .source import HarnessSource, ReplaySource, Task, load_tasks
from .state import Sidecar
from .validate import DeepVerifier, TierAValidator, Verdict, load_schema, make_validator

__all__ = [
    "AppendError",
    "Budget",
    "CommonMessage",
    "DEFAULTS",
    "DeepVerifier",
    "GenericAdapter",
    "HarnessSource",
    "IntegrityReport",
    "IntegritySchedule",
    "JsonlAppender",
    "OpenCodeAdapter",
    "RawTrajectory",
    "Reporter",
    "ReplaySource",
    "Sidecar",
    "Task",
    "TierAValidator",
    "ToolShuffler",
    "Verdict",
    "build_record",
    "load_config",
    "load_schema",
    "load_tasks",
    "make_validator",
    "save_config",
    "verify",
]
