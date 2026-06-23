import calendar
import json
from datetime import date, timedelta
from pathlib import Path
from typing import cast

from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.tx_qa.domain import Direction, Label
from bot.models.tx_qa.query import (
    TimeframeType,
    TXQAParseIntermediateResult,
    TxQAQuery,
)
from bot.models.tx_qa.timeframes import NAMED_TIMEFRAMES, RELATIVE_TIMEFRAMES, DateRange
from bot.routes.tx_qa.timeframe_parse_prompt_builder import (
    DateParserPromptInput,
    build_date_parser_system_prompt,
    build_date_parser_user_prompt,
    load_date_parser_instructions,
)


def parse_intermediate_result(intermediate_raw_result: str) -> TxQAQuery | None:
    try:
        intermediate_result = _deserialize_intermediate_result(intermediate_raw_result)
        date_range = _resolve_date_range(intermediate_result, today=date.today())
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

    if date_range is None:
        return None

    start, end = date_range

    return TxQAQuery(
        label=intermediate_result.label or "other",
        direction=intermediate_result.direction or "spend",
        start=start,
        end=end,
    )

def _deserialize_intermediate_result(raw: str) -> TXQAParseIntermediateResult:
    data = json.loads(raw)
    raw_label = data.get("label")
    raw_direction = data.get("direction")
    return TXQAParseIntermediateResult(
        label=cast(Label, raw_label) if raw_label in ("food", "pets", "other") else None,
        direction=cast(Direction, raw_direction) if raw_direction in ("spend", "receive") else None,
        timeframe_type=TimeframeType(data["timeframe_type"]),
        relative_offset=data.get("relative_offset"),
        year=data.get("year"),
        month=data.get("month"),
        day=data.get("day"),
        quarter=data.get("quarter"),
        start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else None,
        end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
        confidence=data.get("confidence"),
        reason=data.get("reason"),
    )

def _resolve_date_range(
    intermediate_result: TXQAParseIntermediateResult,
    *,
    today: date,
) -> DateRange | None:
    timeframe_type = intermediate_result.timeframe_type
    
    range: DateRange | None = None

    if timeframe_type == TimeframeType.NAMED_DATE:
        range = _explicit_date(intermediate_result, today)

    elif timeframe_type == TimeframeType.DATE_RANGE:
        range = _explicit_date_range(intermediate_result)

    elif timeframe_type in RELATIVE_TIMEFRAMES:
        range = _relative_date_range(timeframe_type, intermediate_result.relative_offset, today)

    elif timeframe_type in NAMED_TIMEFRAMES:
        range = _named_date_range(intermediate_result, today)

    log_event(trace_id="SASA", event="tx_qa.parse", payload={"intermediate_result": intermediate_result, "range": range})

    return range


def _explicit_date_range(intermediate_result: TXQAParseIntermediateResult) -> DateRange | None:
    if not intermediate_result.start_date or not intermediate_result.end_date:
        return None

    return intermediate_result.start_date, intermediate_result.end_date

def _explicit_date(intermediate_result: TXQAParseIntermediateResult, today: date) -> DateRange | None:
    day = intermediate_result.day
    month = intermediate_result.month
    year = intermediate_result.year

    # If year isn't provided, choose the most recent occurrence of the month/day.
    # If the month/day this year is on-or-before `today`, use this year; otherwise use previous year.
    if year is None:
        if not (month and day):
            return None
        try:
            candidate = date(today.year, month, day)
        except ValueError:
            return None

        year = today.year if candidate <= today else today.year - 1

    # Validate final date
    try:
        return _day_range(year, month, day) if year and month and day else None
    except Exception:
        return None

def _relative_date_range(
    timeframe_type: TimeframeType,
    offset: int | None,
    today: date,
) -> DateRange | None:
    if offset is None:
        return None

    match timeframe_type:
        case TimeframeType.RELATIVE_DAY:
            target_day = today + timedelta(days=offset)
            return target_day, target_day

        case TimeframeType.RELATIVE_WEEK:
            current_monday = today - timedelta(days=today.weekday())
            start = current_monday + timedelta(weeks=offset)
            return start, start + timedelta(days=6)

        case TimeframeType.RELATIVE_MONTH:
            month_offset = today.month - 1 + offset
            year = today.year + month_offset // 12
            month = month_offset % 12 + 1
            return _month_range(year, month)

        case TimeframeType.RELATIVE_YEAR:
            return _year_range(today.year + offset)

        case _:
            return None


def _named_date_range(intermediate_result: TXQAParseIntermediateResult, today: date) -> DateRange | None:
    year = intermediate_result.year or today.year

    match intermediate_result.timeframe_type:
        case TimeframeType.NAMED_YEAR:
            return _year_range(year)

        case TimeframeType.NAMED_MONTH:
            if intermediate_result.month is None:
                return None
            return _month_range(year, intermediate_result.month)

        case TimeframeType.NAMED_QUARTER:
            if intermediate_result.quarter is None:
                return None
            return _quarter_range(year, intermediate_result.quarter)

        case TimeframeType.NAMED_DAY:
            if intermediate_result.month is None or intermediate_result.day is None:
                return None

            target_day = date(
                year,
                intermediate_result.month,
                intermediate_result.day,
            )
            return target_day, target_day

        case _:
            return None


def _year_range(year: int) -> DateRange:
    return date(year, 1, 1), date(year, 12, 31)


def _month_range(year: int, month: int) -> DateRange:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)

def _day_range(year: int, month: int, day: int) -> DateRange:
    return date(year, month, day), date(year, month, day)

def _quarter_range(year: int, quarter: int) -> DateRange:
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid quarter: {quarter}")

    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3

    start = date(year, start_month, 1)
    end = date(year, end_month, calendar.monthrange(year, end_month)[1])

    return start, end


def parse_query(session_id: str, llm_client: LLMClient, msg: str) -> TxQAQuery | None:

    prompt_template = load_date_parser_instructions(Path("src/bot/prompts/timeframe_parse_instructions.txt"))
    system_prompt = build_date_parser_system_prompt(
        template=prompt_template,
    )
    user_prompt = build_date_parser_user_prompt(DateParserPromptInput(message=msg))
    
    log_event(
        trace_id=session_id,
        event="tx_qa.date_parser.input",
        payload={"message": msg}
    )
    
    date_parser_response = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt,
    )

    log_event(
        trace_id=session_id,
        event="tx_qa.date_parser.output",
        payload={"date_parser_response": date_parser_response}
    )
    
    parsed = parse_intermediate_result(intermediate_raw_result=date_parser_response)
    if parsed is None:
        log_event(
            trace_id=session_id,
            event="tx_qa.date_parser.parse_failed",
            payload={"date_parser_response": date_parser_response}
        )
    return parsed