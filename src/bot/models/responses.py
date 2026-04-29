from __future__ import annotations

from dataclasses import dataclass

from bot.models.repository import DocReference


@dataclass(frozen=True)
class BotResponse:
    trace_id: str
    answer: str
    doc_references: list[DocReference] # List of DocReference objects
