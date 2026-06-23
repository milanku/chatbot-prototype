from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from bot.models.tx_qa.query import TxQAQuery


@dataclass(frozen=True)
class TxQAQueryResult:
    query: TxQAQuery
    total: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(datetime.timezone.utc))