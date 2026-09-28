from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.tx_qa.memory.models import SessionState
from bot.tx_qa.tx_summary.explain_tx_summary_coordinator import (
    ExplainTxSummaryCoordinator,
)


class ExplainTxSummaryHandler(RouteHandler):
    def __init__(self, *, coordinator: ExplainTxSummaryCoordinator) -> None:
        self._coordinator = coordinator

    def handle(self, *, message: str, session_state: SessionState) -> RouteHandlerResult:
        coordinator_result = self._coordinator.answer(question=message, session_state=session_state)

        return RouteHandlerResult(
            answer_text=coordinator_result.answer,
        )
