from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Final

from pydantic import BaseModel

from bot.models.tx_qa.domain import Direction, Label


@dataclass(frozen=True)
class TxQAQuery:
    label: Label | None
    start: date
    end: date
    direction: Direction | None
    
DateRange = tuple[date, date]

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

class RawRangeEndpoint(BaseModel):
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
        
class RollingRangeUnit(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class RollingRangeMode(Enum):
    TRAILING = "trailing"
    PREVIOUS_COMPLETE = "previous_complete"
 
class TXQARawQuery(BaseModel):
    label: Label | None = None
    direction: Direction | None = None
    timeframe_type: TimeframeType | None = None
    
    # For relative timeframes, e.g., "last month" -> relative_offset = -1, "next month" -> relative_offset = 1
    relative_offset: int | None = None

    # For named timeframes
    year: int | None =  None
    quarter: int | None = None
    month: int | None = None
    day: int | None = None

    # For DATE_RANGE timeframe this contains start date data only.
    start_endpoint: RawRangeEndpoint | None = None
    # For DATE_RANGE timeframe this contains end date data only.
    end_endpoint: RawRangeEndpoint | None = None
    
    # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> mode = RollingRangeMode.TRAILING
    mode: RollingRangeMode | None = None
    # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> unit = RollingRangeUnit.MONTH
    unit: RollingRangeUnit | None = None
    # For ROLLING_RANGE timeframes, e.g., "last 3 months" -> unit_amount = 3
    unit_amount: int | None = None

class TXQAQueryExtraction(BaseModel):
    raw_query_data: TXQARawQuery
    # Confidence score for the parsed timeframe information, between 0 and 1
    confidence: float | None = None
    # Optional reason or explanation for the parsed timeframe information, can be used for debugging or logging purposes
    reason: str | None = None
    
@dataclass(frozen=True)
class TXExplainParseIntermediateResult:
    reference_offset: int | None
    reference_count: int | None
    confidence: float | None
    reason: str | None