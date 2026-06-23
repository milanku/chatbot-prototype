from dataclasses import dataclass

from bot.models.memory import SessionState
from bot.models.repository import DocReference


@dataclass(frozen=True)
class HandlerResult:
    answer_text: str
    new_state: SessionState
    references: list[DocReference]