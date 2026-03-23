from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from bot.routes.tx_qa.parse import TxQAQuery


@dataclass(frozen=True)
class TxQAQueryResult:
    query: TxQAQuery
    total: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SessionState:
    txs_results: tuple[TxQAQueryResult, ...] = field(default_factory=tuple)
