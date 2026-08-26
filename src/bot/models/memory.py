from dataclasses import dataclass

from bot.models.tx_qa.results import SummaryQueryResult


@dataclass(frozen=True)
class SessionState:
    txs_results: tuple[SummaryQueryResult, ...] = ()
    
    def add_tx_result(self, tx_result: SummaryQueryResult) -> "SessionState":
        return SessionState(txs_results=self.txs_results + (tx_result,))
