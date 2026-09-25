from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState


class OutOfScopeHandler(RouteHandler):
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        answer_text = "Sorry, I can't help with that.\n"
        return RouteHandlerResult(
            answer_text=answer_text,
            references=[],
        )
