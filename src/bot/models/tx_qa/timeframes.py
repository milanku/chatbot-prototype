from datetime import date
from enum import Enum

DateRange = tuple[date, date]

class TimeframeType(Enum):
    RELATIVE_DAY = "RELATIVE_DAY"
    RELATIVE_WEEK = "RELATIVE_WEEK"
    RELATIVE_MONTH = "RELATIVE_MONTH"
    RELATIVE_YEAR = "RELATIVE_YEAR"
    NAMED_DATE = "NAMED_DATE"
    NAMED_DAY = "NAMED_DAY"
    NAMED_MONTH = "NAMED_MONTH"
    NAMED_QUARTER = "NAMED_QUARTER"
    NAMED_YEAR = "NAMED_YEAR"
    DATE_RANGE = "DATE_RANGE"
    UNKNOWN = "UNKNOWN"
    
RELATIVE_TIMEFRAMES = {
    TimeframeType.RELATIVE_DAY,
    TimeframeType.RELATIVE_WEEK,
    TimeframeType.RELATIVE_MONTH,
    TimeframeType.RELATIVE_YEAR,
}

NAMED_TIMEFRAMES = {
    TimeframeType.NAMED_DATE,
    TimeframeType.NAMED_DAY,
    TimeframeType.NAMED_MONTH,
    TimeframeType.NAMED_QUARTER,
    TimeframeType.NAMED_YEAR,
}