from dataclasses import dataclass

from bot.models.tx_qa.results import SummaryQueryResult


@dataclass(frozen=True)
class SessionState:
    tx_summaries: tuple[SummaryQueryResult, ...] = ()
    
    def add_tx_summary(self, tx_summary: SummaryQueryResult) -> "SessionState":
        return SessionState(tx_summaries=self.tx_summaries + (tx_summary,))
