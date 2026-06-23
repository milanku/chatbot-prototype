from bot.handlers.models import HandlerResult


class OutOfScopeHandler:
    def handle(self, *, message:str, session_id:str, session_state, trace)  -> HandlerResult:
        answer_text = "Sorry, I can't help with that.\n"
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )