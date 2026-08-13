from datetime import date
from pathlib import Path

from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.tx_qa.query import (
    TxQAQuery,
    TXQAQueryExtraction,
)
from bot.routes.tx_qa.timeframe.resolver import resolve_date_range_from_raw_query
from bot.routes.tx_qa.timeframe_parse_prompt_builder import (
    TimeframeParserPromptInput,
    build_timeframe_parser_system_prompt,
    build_timeframe_parser_user_prompt,
    load_timeframe_parser_instructions,
)


def parse_tx_query_from_user_message(llm_client: LLMClient, user_msg: str) -> TxQAQuery | None:
    prompt_template = load_timeframe_parser_instructions(Path("src/bot/prompts/timeframe_parse_instructions.txt"))
    system_prompt = build_timeframe_parser_system_prompt(template=prompt_template)
    user_prompt = build_timeframe_parser_user_prompt(TimeframeParserPromptInput(message=user_msg))
    
    log_event(
        event="tx_qa.timeframe_parser.input",
        payload={"message": user_msg}
    )
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=TXQAQueryExtraction,
        system_instructions=system_prompt,
    )

    log_event(
        event="tx_qa.timeframe_parser.output",
        payload={"llm_response": llm_structured_response}
    )
    
    date_range = resolve_date_range_from_raw_query(llm_structured_response.raw_query_data, today=date.today())
    if date_range is None:
        return None

    start, end = date_range

    return TxQAQuery(
        label=llm_structured_response.raw_query_data.label,
        direction=llm_structured_response.raw_query_data.direction,
        start=start,
        end=end,
    )