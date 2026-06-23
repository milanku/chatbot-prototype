from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState


@dataclass(frozen=True)
class HandlerResult:
    answer_text: str
    new_state: SessionState
    references: list[DocReference]