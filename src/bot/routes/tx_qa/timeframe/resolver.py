from datetime import date

from bot.logging import log_event
from bot.models.tx_qa.query import (
    DateRange,
    RollingRangeMode,
    TimeframeType,
    TXQAParseRawQueryData,
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


def resolve_date_range_from_raw_query_data(
    raw_query_data: TXQAParseRawQueryData,
    *,
    today: date,
) -> DateRange | None:
    
    timeframe_type = raw_query_data.timeframe_type
            
    resolved_range: DateRange | None = None

    match timeframe_type:
        case TimeframeType.NAMED_YEAR:
            if(raw_query_data.year is not None):
                resolved_range = resolve_named_year(year=raw_query_data.year)
        case TimeframeType.NAMED_MONTH:
            if (raw_query_data.month is not None):
                resolved_range = resolve_named_month(
                    year=raw_query_data.year,
                    month=raw_query_data.month,
                    today=today
                )
        case TimeframeType.NAMED_QUARTER:
            if (raw_query_data.quarter is not None):
                resolved_range = resolve_named_quarter(
                    year=raw_query_data.year,
                    quarter=raw_query_data.quarter,
                    today=today
                )
        case TimeframeType.NAMED_WEEKDAY:
            if (raw_query_data.day is not None):
                resolved_range = resolve_named_weekday(
                    weekday=raw_query_data.day,
                    offset=raw_query_data.relative_offset,
                    today=today
                )
        case TimeframeType.NAMED_DATE:
            if (raw_query_data.day is not None):
                resolved_range = resolve_named_date(
                    year=raw_query_data.year,
                    month=raw_query_data.month,
                    day=raw_query_data.day,
                    today=today
                )
        case TimeframeType.RELATIVE_YEAR:
            if (raw_query_data.relative_offset is not None):
                resolved_range = resolve_relative_year(
                    offset=raw_query_data.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_MONTH:
            if (raw_query_data.relative_offset is not None):
                resolved_range = resolve_relative_month(
                    offset=raw_query_data.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_WEEK:
            if (raw_query_data.relative_offset is not None):
                resolved_range = resolve_relative_week(
                    offset=raw_query_data.relative_offset,
                    today=today
                )
        case TimeframeType.RELATIVE_DAY:
            if (raw_query_data.relative_offset is not None):
                resolved_range = resolve_relative_day(
                    offset=raw_query_data.relative_offset,
                    today=today
                )
        case TimeframeType.EXPLICIT_RANGE:
            if raw_query_data.start_endpoint is not None and raw_query_data.start_endpoint.has_data and raw_query_data.end_endpoint is not None and raw_query_data.end_endpoint.has_data:
                resolved_range = resolve_explicit_range(
                    start_endpoint=raw_query_data.start_endpoint,
                    end_endpoint=raw_query_data.end_endpoint,
                    today=today
                )
        case TimeframeType.ROLLING_RANGE:
            if(raw_query_data.unit is not None and raw_query_data.amount is not None):
                if(raw_query_data.mode == RollingRangeMode.TRAILING):
                    resolved_range = resolve_trailing_range(
                        unit=raw_query_data.unit,
                        amount=raw_query_data.amount,
                        today=today
                    )
                elif(raw_query_data.mode == RollingRangeMode.PREVIOUS_COMPLETE):
                    resolved_range = resolve_previous_complete_range(
                        unit=raw_query_data.unit,
                        amount=raw_query_data.amount,
                        today=today
                    )
        case _:
            resolved_range = None

    log_event(event="tx_qa.parse", payload={"raw_query_data": raw_query_data, "range": resolved_range})

    return resolved_range