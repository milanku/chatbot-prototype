from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal

from bot.models.tx_qa.domain import Direction, Label
from bot.models.tx_qa.timeframes import TimeframeType


@dataclass(frozen=True)
class TxFilter:
    label: Label
    start: date
    end: date
    direction: Direction

@dataclass(frozen=True)
class TxQAQuery:
    label: Label
    start: date
    end: date
    direction: Direction
    
@dataclass(frozen=True)
class TxQAQueryResult:
    query: TxQAQuery
    total: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass(frozen=True)
class TXQAParseIntermediateResult:
    label: Label | None
    direction: Direction | None
    timeframe_type: TimeframeType
    relative_offset: int | None # For relative timeframes, e.g., "last month" -> relative_offset = -1, "next month" -> relative_offset = 1
    year: int | None # For named timeframes, e.g., "January 2026" -> year = 2026
    month: int | None # For named timeframes, e.g., "January 2026" -> month = 1
    day: int | None # For named timeframes, e.g., "January 1, 2026" -> day = 1
    quarter: int | None # For named timeframes, e.g., "Q1 2026" -> quarter = 1
    start_date: date | None # For date range timeframes, e.g., "from 2026-01-01 to 2026-01-31" -> start_date = 2026-01-01
    end_date: date | None # For date range timeframes, e.g., "from 2026-01-01 to 2026-01-31" -> end_date = 2026-01-31
    confidence: float | None # Confidence score for the parsed timeframe information, between 0 and 1
    reason: str | None # Optional reason or explanation for the parsed timeframe information, can be used for debugging or logging purposes
    
@dataclass(frozen=True)
class TXExplainParseIntermediateResult:
    reference_offset: int | None
    reference_count: int | None
    confidence: float | None
    reason: str | None