from abc import ABC, abstractmethod

from pydantic.dataclasses import dataclass

from bot.doc_qa.indexing.models import DocReference
from bot.tx_qa.memory.models import SessionState


@dataclass(frozen=True, kw_only=True)
class RouteHandlerResult:
    answer_text: str
    new_state: SessionState | None = None

    def result_to_str(self) -> str:
        return self.answer_text


@dataclass(frozen=True)
class DocQARouterHandlerResult(RouteHandlerResult):
    doc_references: list[DocReference]

    def _referenced_docs_to_str(self) -> str:
        return "\n".join(
            f"{ref.file_name} ({' >> '.join(ref.heading_path)})" for ref in self.doc_references
        )

    def result_to_str(self) -> str:
        referenced_docs = (
            "\n\nReferenced documents:\n" + self._referenced_docs_to_str() + "\n\n"
            if self.doc_references
            else ""
        )
        return self.answer_text + referenced_docs


class RouteHandler(ABC):
    @abstractmethod
    def handle(
        self,
        *,
        message: str,
        session_state: SessionState,
    ) -> RouteHandlerResult:
        raise NotImplementedError
