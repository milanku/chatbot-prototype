from bot.doc_qa.coordinator import DocsAnswerCoordinator
from bot.doc_qa.models import DocsAnswerResult
from bot.handlers.models import DocQARouterHandlerResult, RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState


class DocsAnswerHandler(RouteHandler):
    def __init__(self, *, coordinator: DocsAnswerCoordinator) -> None:
        self._coordinator = coordinator

    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        answer: DocsAnswerResult = self._coordinator.answer(
            question=message,
        )

        # FUTURE: Add translations/localization for answer.status values and supported languages

        return DocQARouterHandlerResult(
            answer_text=answer.answer_text,
            doc_references=answer.doc_references,
        )
