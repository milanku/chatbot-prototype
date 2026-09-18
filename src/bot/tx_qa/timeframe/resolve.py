from datetime import date

from bot.tx_qa.timeframe.explicit_range import resolve_explicit_range
from bot.tx_qa.timeframe.models import (
    DateRange,
    RollingRangeMode,
    Timeframe,
    TimeframeType,
)
from bot.tx_qa.timeframe.named import (
    resolve_named_date,
    resolve_named_month,
    resolve_named_quarter,
    resolve_named_year,
)
from bot.tx_qa.timeframe.relative import (
    resolve_relative_day,
    resolve_relative_month,
    resolve_relative_week,
    resolve_relative_year,
)
from bot.tx_qa.timeframe.rolling import (
    resolve_previous_complete_range,
    resolve_trailing_range,
)
from bot.tx_qa.timeframe.weekday import resolve_named_weekday


def resolve_date_range(
    timeframe: Timeframe,
    today: date,
) -> DateRange | None:
    timeframe_type = timeframe.timeframe_type
    resolved_range: DateRange | None = None

    match timeframe_type:
        case TimeframeType.NAMED_YEAR:
            if(timeframe.year is not None):
                resolved_range = resolve_named_year(year=timeframe.year)
        case TimeframeType.NAMED_MONTH:
            if (timeframe.month is not None):
                resolved_range = resolve_named_month(
                    year=timeframe.year,
                    month=timeframe.month,
                    today=today
                )
        case TimeframeType.NAMED_QUARTER:
            if (timeframe.quarter is not None):
                resolved_range = resolve_named_quarter(
                    year=timeframe.year,
                    quarter=timeframe.quarter,
                    today=today
                )
        case TimeframeType.NAMED_WEEKDAY:
            if (timeframe.day is not None):
                resolved_range = resolve_named_weekday(
                    weekday=timeframe.day,
                    offset=timeframe.relative_offset,
                    today=today
                )
        case TimeframeType.NAMED_DATE:
            if (timeframe.day is not None):
                resolved_range = resolve_named_date(
                    year=timeframe.year,
                    month=timeframe.month,
                    day=timeframe.day,
                    today=today
                )
        case TimeframeType.RELATIVE_YEAR:
            if (timeframe.relative_offset is not None):
                resolved_range = resolve_relative_year(
                    offset=timeframe.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_MONTH:
            if (timeframe.relative_offset is not None):
                resolved_range = resolve_relative_month(
                    offset=timeframe.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_WEEK:
            if (timeframe.relative_offset is not None):
                resolved_range = resolve_relative_week(
                    offset=timeframe.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_DAY:
            if (timeframe.relative_offset is not None):
                resolved_range = resolve_relative_day(
                    offset=timeframe.relative_offset,
                    today=today
                )
        case TimeframeType.EXPLICIT_RANGE:
            if timeframe.start_endpoint is not None and timeframe.start_endpoint.has_data and timeframe.end_endpoint is not None and timeframe.end_endpoint.has_data:
                resolved_range = resolve_explicit_range(
                    start_endpoint=timeframe.start_endpoint,
                    end_endpoint=timeframe.end_endpoint,
                    today=today
                )
        case TimeframeType.ROLLING_RANGE:
            if(timeframe.unit is not None and timeframe.unit_amount is not None):
                if(timeframe.mode == RollingRangeMode.TRAILING):
                    resolved_range = resolve_trailing_range(
                        unit=timeframe.unit,
                        unit_amount=timeframe.unit_amount,
                        today=today
                    )
                elif(timeframe.mode == RollingRangeMode.PREVIOUS_COMPLETE):
                    resolved_range = resolve_previous_complete_range(
                        unit=timeframe.unit,
                        unit_amount=timeframe.unit_amount,
                        today=today
                    )
        case _:
            resolved_range = None

    return resolved_range