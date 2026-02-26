from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from bot.models.domain import Transaction
from bot.recipes.tx_qa.parse import TxQAQuery


@dataclass(frozen=True)
class TxQAQueryResult:
    query: TxQAQuery
    txs: list[Transaction]
    total: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionState:
    txs_results: list[TxQAQueryResult] = field(default_factory=list)
