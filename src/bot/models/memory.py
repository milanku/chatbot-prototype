from dataclasses import dataclass

from bot.models.tx_qa.results import TxQAQueryResult


@dataclass(frozen=True)
class SessionState:
    txs_results: tuple[TxQAQueryResult, ...] = ()
    
    def add_tx_result(self, tx_result: TxQAQueryResult) -> "SessionState":
        return SessionState(txs_results=self.txs_results + (tx_result,))
