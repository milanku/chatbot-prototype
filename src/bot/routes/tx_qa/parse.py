import json
from datetime import date
from pathlib import Path

from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.tx_qa.query import (
    TxQAQuery,
)
from bot.routes.tx_qa.deserialize import deserialize_raw_query_data
from bot.routes.tx_qa.timeframe.resolver import resolve_date_range_from_raw_query_data
from bot.routes.tx_qa.timeframe_parse_prompt_builder import (
    TimeframeParserPromptInput,
    build_timeframe_parser_system_prompt,
    build_timeframe_parser_user_prompt,
    load_timeframe_parser_instructions,
)


def get_tx_query_from_serialized_raw_query_data(serialized_raw_query_data: str) -> TxQAQuery | None:
    try:
        raw_query_data = deserialize_raw_query_data(serialized_raw_query_data)
        date_range = resolve_date_range_from_raw_query_data(raw_query_data, today=date.today())
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

    if date_range is None:
        return None

    start, end = date_range

    return TxQAQuery(
        label=raw_query_data.label,
        direction=raw_query_data.direction,
        start=start,
        end=end,
    )

def parse_tx_query_from_user_message(llm_client: LLMClient, user_msg: str) -> TxQAQuery | None:
    prompt_template = load_timeframe_parser_instructions(Path("src/bot/prompts/timeframe_parse_instructions.txt"))
    system_prompt = build_timeframe_parser_system_prompt(template=prompt_template)
    user_prompt = build_timeframe_parser_user_prompt(TimeframeParserPromptInput(message=user_msg))
    
    log_event(
        event="tx_qa.timeframe_parser.input",
        payload={"message": user_msg}
    )
    
    serialized_raw_query_data = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt,
    )

    log_event(
        event="tx_qa.timeframe_parser.output",
        payload={"serialized_raw_query_data": serialized_raw_query_data}
    )
    
    parsed_query = get_tx_query_from_serialized_raw_query_data(serialized_raw_query_data=serialized_raw_query_data)
    if parsed_query is None:
        log_event(
            event="tx_qa.timeframe_parser.parse_failed",
            payload={"serialized_raw_query_data": serialized_raw_query_data}
        )
    return parsed_query