from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState


class UnknownRouteHandler(RouteHandler): 
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        return RouteHandlerResult(   
            answer_text="Unknown route.\n",
            references=[],
        )