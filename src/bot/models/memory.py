from dataclasses import dataclass, field

from bot.models.tx_qa.query import TxQAQueryResult


@dataclass(frozen=True)
class SessionState:
    txs_results: tuple[TxQAQueryResult, ...] = field(default_factory=tuple)
