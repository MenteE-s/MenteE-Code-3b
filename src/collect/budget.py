"""Run budgets: stop the collector before it becomes expensive.

Three independent ceilings, whichever is hit first wins. A budget that cannot
be exceeded is a budget that does not need to be watched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["Budget"]


@dataclass
class Budget:
    max_rows: int | None = None
    max_minutes: float | None = None
    max_generation_tokens: int | None = None

    def __post_init__(self) -> None:
        self.started = time.time()
        self.rows = 0
        self.tokens = 0

    # ------------------------------------------------------------------ state
    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started

    def consume(self, rows: int = 0, tokens: int = 0) -> None:
        self.rows += rows
        self.tokens += tokens

    # ------------------------------------------------------------------ checks
    def exhausted(self) -> str | None:
        """Return the name of the exhausted budget, or None."""
        if self.max_rows is not None and self.rows >= self.max_rows:
            return "max_rows"
        if self.max_minutes is not None and self.elapsed_s / 60.0 >= self.max_minutes:
            return "max_minutes"
        if self.max_generation_tokens is not None and self.tokens >= self.max_generation_tokens:
            return "max_generation_tokens"
        return None

    def summary(self) -> dict[str, float | int | None]:
        return {
            "rows": self.rows,
            "tokens": self.tokens,
            "elapsed_s": round(self.elapsed_s, 1),
            "max_rows": self.max_rows,
            "max_minutes": self.max_minutes,
            "max_generation_tokens": self.max_generation_tokens,
        }
