from datetime import datetime

from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.models.memory import SessionState
from bot.routes.tx_qa.tx_summary_coordinator import TxSummaryCoordinator


class TxSummaryHandler(RouteHandler):
    def __init__(self, *, coordinator: TxSummaryCoordinator) -> None:
        self._coordinator = coordinator
        
    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        coordinator_result = self._coordinator.answer(message, session_state=session_state, today=datetime.now())

        return RouteHandlerResult(
            answer_text=coordinator_result.answer_text,
            new_state=coordinator_result.new_state,
            references=[],
        )