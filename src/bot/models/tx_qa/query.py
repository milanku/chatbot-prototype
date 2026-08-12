from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Final

from bot.models.tx_qa.domain import Direction, Label


@dataclass(frozen=True)
class TxQAQuery:
    label: Label | None
    start: date
    end: date
    direction: Direction | None
    
DateRange = tuple[date, date]

@dataclass(frozen=True)
class RawRangeEndpoint:
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    day: int | None = None

    @property
    def has_data(self) -> bool:
        return any(
            value is not None
            for value in (
                self.year,
                self.quarter,
                self.month,
                self.day,
            )
        )

class TimeframeType(Enum):
    RELATIVE_DAY = "RELATIVE_DAY"
    RELATIVE_WEEK = "RELATIVE_WEEK"
    RELATIVE_MONTH = "RELATIVE_MONTH"
    RELATIVE_YEAR = "RELATIVE_YEAR"
    NAMED_DATE = "NAMED_DATE"
    NAMED_WEEKDAY = "NAMED_WEEKDAY"
    NAMED_MONTH = "NAMED_MONTH"
    NAMED_QUARTER = "NAMED_QUARTER"
    NAMED_YEAR = "NAMED_YEAR"
    EXPLICIT_RANGE = "EXPLICIT_RANGE"
    ROLLING_RANGE = "ROLLING_RANGE"
    UNKNOWN = "UNKNOWN"

RELATIVE_TIMEFRAMES: Final[frozenset[TimeframeType]] = frozenset(
    {
        TimeframeType.RELATIVE_DAY,
        TimeframeType.RELATIVE_WEEK,
        TimeframeType.RELATIVE_MONTH,
        TimeframeType.RELATIVE_YEAR,
    }
)

NAMED_RANGE_TIMEFRAMES: Final[frozenset[TimeframeType]] = frozenset(
    {
        TimeframeType.NAMED_WEEKDAY,
        TimeframeType.NAMED_MONTH,
        TimeframeType.NAMED_QUARTER,
        TimeframeType.NAMED_YEAR,
    }
)

class RollingRangeUnit(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class RollingRangeMode(Enum):
    TRAILING = "trailing"
    PREVIOUS_COMPLETE = "previous_complete"

@dataclass(frozen=True)
class TXQAParseRawQueryData:
    label: Label | None
    direction: Direction | None
    timeframe_type: TimeframeType
    relative_offset: int | None # For relative timeframes, e.g., "last month" -> relative_offset = -1, "next month" -> relative_offset = 1
    year: int | None # For named timeframes, e.g., "January 2026" -> year = 2026
    quarter: int | None # For named timeframes, e.g., "Q1 2026" -> quarter = 1
    month: int | None # For named timeframes, e.g., "January 2026" -> month = 1
    day: int | None # For named timeframes, e.g., "January 1, 2026" -> day = 1
    start_endpoint: RawRangeEndpoint | None # For DATE_RANGE timeframe this contains start date data only.
    end_endpoint: RawRangeEndpoint | None # For DATE_RANGE timeframe this contains end date data only.
    unit: RollingRangeUnit | None # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> unit = RollingRangeUnit.MONTH
    mode: RollingRangeMode | None # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> mode = RollingRangeMode.TRAILING
    amount: int | None # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> amount = 3
    confidence: float | None # Confidence score for the parsed timeframe information, between 0 and 1
    reason: str | None # Optional reason or explanation for the parsed timeframe information, can be used for debugging or logging purposes
    
@dataclass(frozen=True)
class TXExplainParseIntermediateResult:
    reference_offset: int | None
    reference_count: int | None
    confidence: float | None
    reason: str | None