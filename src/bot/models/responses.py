from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BotResponse:
    answer: str
    references: list[str]
    trace_id: str
