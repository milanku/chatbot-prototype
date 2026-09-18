from datetime import date

from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState
from bot.tx_qa.tx_list.coordinator import TxListCoordinator


class TxListHandler(RouteHandler):    
    def __init__(self, *, coordinator: TxListCoordinator) -> None:
        self._coordinator = coordinator
    
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        tx_list_answer = self._coordinator.answer(question=message, today=date.today())
        
        return RouteHandlerResult(
            answer_text=tx_list_answer.answer,
            references=[],
        )