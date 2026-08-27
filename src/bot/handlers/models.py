from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState


@dataclass(frozen=True)
class RouteHandlerResult:
    answer_text: str
    references: list[DocReference]
    new_state: SessionState | None = None
    
class RouteHandler(ABC):
    @abstractmethod
    def handle(
        self,
        *,
        message: str,
        session_state: SessionState,
    ) -> RouteHandlerResult:
        raise NotImplementedError