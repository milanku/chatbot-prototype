from datetime import date

from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.models.memory import SessionState
from bot.routes.tx_qa.tx_list_coordinator import TxListCoordinator


class TxListHandler(RouteHandler):    
    def __init__(self, *, coordinator: TxListCoordinator) -> None:
        self._coordinator = coordinator
    
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        tx_list_answer = self._coordinator.answer(question=message, today=date.today())
        
        return RouteHandlerResult(
            answer_text=tx_list_answer.answer,
            new_state=session_state,
            references=[],
        )