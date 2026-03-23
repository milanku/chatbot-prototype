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


def parse_intermediate_result(intermediate_raw_result: str) -> TxQAQuery | None:
    try:
        intermediate_result = _deserialize_intermediate_result(intermediate_raw_result)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    current_date = date.today()
    
    if intermediate_result.timeframe_type == TimeframeType.DATE_RANGE:
        if intermediate_result.start_date and intermediate_result.end_date:
            return TxQAQuery(
                label=intermediate_result.label or "other",
                direction=intermediate_result.direction or "spend",
                start=intermediate_result.start_date,
                end=intermediate_result.end_date,
            )
        else:
            return None
    if intermediate_result.timeframe_type in [TimeframeType.RELATIVE_DAY, TimeframeType.RELATIVE_WEEK, TimeframeType.RELATIVE_MONTH, TimeframeType.RELATIVE_YEAR]:
        if intermediate_result.relative_offset is not None:
            if intermediate_result.timeframe_type == TimeframeType.RELATIVE_DAY:
                start = current_date + timedelta(days=intermediate_result.relative_offset)
                end = start
            elif intermediate_result.timeframe_type == TimeframeType.RELATIVE_WEEK:
                current_monday = current_date - timedelta(days=current_date.weekday())
                start = current_monday + timedelta(weeks=intermediate_result.relative_offset)
                end = start + timedelta(days=6)
            elif intermediate_result.timeframe_type == TimeframeType.RELATIVE_MONTH:
                month_offset = (current_date.month - 1) + intermediate_result.relative_offset
                year_offset = month_offset // 12
                month = (month_offset % 12) + 1
                year = current_date.year + year_offset
                start = date(year, month, 1)
                end = date(year, month, calendar.monthrange(year, month)[1])
            elif intermediate_result.timeframe_type == TimeframeType.RELATIVE_YEAR:
                year = current_date.year + intermediate_result.relative_offset
                start = date(year, 1, 1)
                end = date(year, 12, 31)
            else:
                return None
            
            return TxQAQuery(
                label=intermediate_result.label or "other",
                direction=intermediate_result.direction or "spend",
                start=start,
                end=end,
            )
    if intermediate_result.timeframe_type in [TimeframeType.NAMED_DAY, TimeframeType.NAMED_MONTH, TimeframeType.NAMED_QUARTER, TimeframeType.NAMED_YEAR]:
        year = intermediate_result.year if intermediate_result.year is not None else current_date.year
        if intermediate_result.timeframe_type == TimeframeType.NAMED_YEAR:
            start = date(year, 1, 1)
            end = date(year, 12, 31)
        elif intermediate_result.timeframe_type == TimeframeType.NAMED_MONTH and intermediate_result.month is not None:
            month = intermediate_result.month
            start = date(year, month, 1)
            end = date(year, month, calendar.monthrange(year, month)[1])
        elif intermediate_result.timeframe_type == TimeframeType.NAMED_QUARTER and intermediate_result.quarter is not None:
            quarter = intermediate_result.quarter
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3
            start = date(year, start_month, 1)
            end = date(year, end_month, calendar.monthrange(year, end_month)[1])
        elif intermediate_result.timeframe_type == TimeframeType.NAMED_DAY and intermediate_result.month is not None and intermediate_result.day is not None:
            month = intermediate_result.month
            day = intermediate_result.day
            start = date(year, month, day)
            end = start
        else:
            return None

        return TxQAQuery(
            label=intermediate_result.label or "other",
            direction=intermediate_result.direction or "spend",
            start=start,
            end=end,
        )
    return None


def parse_query(session_id: str, llm_client: LLMClient, msg: str) -> TxQAQuery | None:

    prompt_template = load_date_parser_instructions(Path("src/bot/routes/tx_qa/timeframe_parse_instructions.txt"))
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