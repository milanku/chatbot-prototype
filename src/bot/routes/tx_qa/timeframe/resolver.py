from datetime import date

from bot.logging import log_event
from bot.models.tx_qa.query import (
    DateRange,
    RollingRangeMode,
    TimeframeType,
    TXQARawQuery,
)
from bot.routes.tx_qa.timeframe.explicit_range import resolve_explicit_range
from bot.routes.tx_qa.timeframe.named import (
    resolve_named_date,
    resolve_named_month,
    resolve_named_quarter,
    resolve_named_year,
)
from bot.routes.tx_qa.timeframe.relative import (
    resolve_relative_day,
    resolve_relative_month,
    resolve_relative_week,
    resolve_relative_year,
)
from bot.routes.tx_qa.timeframe.rolling import (
    resolve_previous_complete_range,
    resolve_trailing_range,
)
from bot.routes.tx_qa.timeframe.weekday import resolve_named_weekday


def resolve_date_range_from_raw_query(
    raw_query: TXQARawQuery,
    *,
    today: date,
) -> DateRange | None:
    
    timeframe_type = raw_query.timeframe_type
            
    resolved_range: DateRange | None = None

    match timeframe_type:
        case TimeframeType.NAMED_YEAR:
            if(raw_query.year is not None):
                resolved_range = resolve_named_year(year=raw_query.year)
        case TimeframeType.NAMED_MONTH:
            if (raw_query.month is not None):
                resolved_range = resolve_named_month(
                    year=raw_query.year,
                    month=raw_query.month,
                    today=today
                )
        case TimeframeType.NAMED_QUARTER:
            if (raw_query.quarter is not None):
                resolved_range = resolve_named_quarter(
                    year=raw_query.year,
                    quarter=raw_query.quarter,
                    today=today
                )
        case TimeframeType.NAMED_WEEKDAY:
            if (raw_query.day is not None):
                resolved_range = resolve_named_weekday(
                    weekday=raw_query.day,
                    offset=raw_query.relative_offset,
                    today=today
                )
        case TimeframeType.NAMED_DATE:
            if (raw_query.day is not None):
                resolved_range = resolve_named_date(
                    year=raw_query.year,
                    month=raw_query.month,
                    day=raw_query.day,
                    today=today
                )
        case TimeframeType.RELATIVE_YEAR:
            if (raw_query.relative_offset is not None):
                resolved_range = resolve_relative_year(
                    offset=raw_query.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_MONTH:
            if (raw_query.relative_offset is not None):
                resolved_range = resolve_relative_month(
                    offset=raw_query.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_WEEK:
            if (raw_query.relative_offset is not None):
                resolved_range = resolve_relative_week(
                    offset=raw_query.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_DAY:
            if (raw_query.relative_offset is not None):
                resolved_range = resolve_relative_day(
                    offset=raw_query.relative_offset,
                    today=today
                )
        case TimeframeType.EXPLICIT_RANGE:
            if raw_query.start_endpoint is not None and raw_query.start_endpoint.has_data and raw_query.end_endpoint is not None and raw_query.end_endpoint.has_data:
                resolved_range = resolve_explicit_range(
                    start_endpoint=raw_query.start_endpoint,
                    end_endpoint=raw_query.end_endpoint,
                    today=today
                )
        case TimeframeType.ROLLING_RANGE:
            if(raw_query.unit is not None and raw_query.unit_amount is not None):
                if(raw_query.mode == RollingRangeMode.TRAILING):
                    resolved_range = resolve_trailing_range(
                        unit=raw_query.unit,
                        unit_amount=raw_query.unit_amount,
                        today=today
                    )
                elif(raw_query.mode == RollingRangeMode.PREVIOUS_COMPLETE):
                    resolved_range = resolve_previous_complete_range(
                        unit=raw_query.unit,
                        unit_amount=raw_query.unit_amount,
                        today=today
                    )
        case _:
            resolved_range = None

    log_event(event="tx_qa.resolve_date_range_from_raw_query", payload={"raw_query_data": raw_query, "range": resolved_range})

    return resolved_range