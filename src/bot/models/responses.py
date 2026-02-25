from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BotResponse:
    answer: str
    references: List[str]
    trace_id: str