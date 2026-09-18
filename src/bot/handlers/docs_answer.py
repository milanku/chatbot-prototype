from bot.doc_qa.coordinator import DocsAnswerCoordinator, DocsAnswerResult
from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState


class DocsAnswerHandler(RouteHandler):
    def __init__(self, *, coordinator: DocsAnswerCoordinator) -> None:
        self._coordinator = coordinator
        
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        answer: DocsAnswerResult = self._coordinator.answer(
            question=message,
        )

        return RouteHandlerResult(
            answer_text=answer.answer_text,
            references=answer.references,
        )