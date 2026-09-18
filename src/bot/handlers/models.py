from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.doc_qa.indexing.models import DocReference
from bot.tx_qa.memory.models import SessionState


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