from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.models.memory import SessionState
from bot.routes.doc_qa.coordinator import DocsAnswer, DocsAnswerCoordinator


class DocsAnswerHandler(RouteHandler):
    def __init__(self, *, coordinator: DocsAnswerCoordinator) -> None:
        self._coordinator = coordinator
        
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        answer: DocsAnswer = self._coordinator.answer(
            question=message,
        )

        return RouteHandlerResult(
            answer_text=answer.answer_text,
            references=answer.references,
        )