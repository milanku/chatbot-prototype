from bot.handlers.models import HandlerResult
from bot.models.memory import SessionState


class UnknownRouteHandler:
    def handle(self, *, message:str, session_id:str, session_state: SessionState)  -> HandlerResult:
        return HandlerResult(   
            answer_text="Unknown route.\n",
            new_state=session_state,
            references=[],
        )