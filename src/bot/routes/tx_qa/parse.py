import calendar
import json
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import cast

from bot.llm.llm_client import LLMClient
from bot.logging import log_event
from bot.models.domain import Direction, Label
from bot.routes.tx_qa.timeframe_parse_prompt_builder import (
    DateParserPromptInput,
    build_date_parser_system_prompt,
    build_date_parser_user_prompt,
    load_date_parser_instructions,
)


@dataclass(frozen=True)
class TxQAQuery:
    label: Label
    start: date
    end: date
    direction: Direction

DateRange = tuple[date, date]

class TimeframeType(Enum):
    RELATIVE_DAY = "RELATIVE_DAY"
    RELATIVE_WEEK = "RELATIVE_WEEK"
    RELATIVE_MONTH = "RELATIVE_MONTH"
    RELATIVE_YEAR = "RELATIVE_YEAR"
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
    TimeframeType.NAMED_DAY,
    TimeframeType.NAMED_MONTH,
    TimeframeType.NAMED_QUARTER,
    TimeframeType.NAMED_YEAR,
}

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

    if timeframe_type == TimeframeType.DATE_RANGE:
        return _explicit_date_range(intermediate_result)

    if timeframe_type in RELATIVE_TIMEFRAMES:
        return _relative_date_range(timeframe_type, intermediate_result.relative_offset, today)

    if timeframe_type in NAMED_TIMEFRAMES:
        return _named_date_range(intermediate_result, today)

    return None


def _explicit_date_range(intermediate_result: TXQAParseIntermediateResult) -> DateRange | None:
    if not intermediate_result.start_date or not intermediate_result.end_date:
        return None

    return intermediate_result.start_date, intermediate_result.end_date


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