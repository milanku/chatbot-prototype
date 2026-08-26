from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.models.memory import SessionState
from bot.routes.tx_qa.tx_explain_summary_coordinator import TxExplainSummaryCoordinator


class TxExplainSummaryHandler(RouteHandler):
    def __init__(self, *, coordinator: TxExplainSummaryCoordinator) -> None:
        self._coordinator = coordinator
    
    def handle(self, *, message: str, session_state: SessionState)  -> RouteHandlerResult:
        coordinator_result = self._coordinator.answer(question=message, session_state=session_state)
        
        return RouteHandlerResult(
            answer_text=coordinator_result.answer,
            new_state=None,
            references=[],
        )