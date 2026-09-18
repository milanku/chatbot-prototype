from dataclasses import dataclass
from typing import Protocol

from bot.tx_qa.tx_summary.models import SummaryQueryResult


@dataclass(frozen=True)
class SessionState:
    tx_summaries: tuple[SummaryQueryResult, ...] = ()
    
    def add_tx_summary(self, tx_summary: SummaryQueryResult) -> "SessionState":
        return SessionState(tx_summaries=self.tx_summaries + (tx_summary,))
    
class SessionStore(Protocol):
    def get_session(self, session_id: str) -> SessionState:
        ...

    def set_session(self, session_id: str, session_state: SessionState) -> None:
        ...