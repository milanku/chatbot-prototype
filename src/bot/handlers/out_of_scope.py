from bot.handlers.base import Handler
from bot.handlers.models import HandlerResult
from bot.models.memory import SessionState


class OutOfScopeHandler(Handler):
    def handle(self, *, message:str, session_id:str, session_state: SessionState)  -> HandlerResult:
        answer_text = "Sorry, I can't help with that.\n"
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )