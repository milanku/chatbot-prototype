from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from bot.models.tx_qa.domain import Transaction
from bot.models.tx_qa.query import TxQuery


@dataclass(frozen=True)
class SummaryQueryResult:
    query: TxQuery
    transactions: list[Transaction]
    total: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))