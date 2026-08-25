from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.models.memory import SessionState


class UnknownRouteHandler(RouteHandler):
    
    def handle(self, *, message: str, session_state: SessionState)  -> RouteHandlerResult:
        return RouteHandlerResult(   
            answer_text="Unknown route.\n",
            new_state=session_state,
            references=[],
        )