from pathlib import Path

from bot.llm.client import LLMClient
from bot.models.tx_qa.query import (
    TXQAQueryExtraction,
)
from bot.routes.tx_qa.timeframe_parse_prompt_builder import (
    TimeframeParserPromptInput,
    build_timeframe_parser_system_prompt,
    build_timeframe_parser_user_prompt,
    load_timeframe_parser_instructions,
)


def parse_raw_tx_query_from_user_message(llm_client: LLMClient, user_msg: str) -> TXQAQueryExtraction:
    prompt_template = load_timeframe_parser_instructions(Path("src/bot/prompts/timeframe_parse_instructions.txt"))
    system_prompt = build_timeframe_parser_system_prompt(template=prompt_template)
    user_prompt = build_timeframe_parser_user_prompt(TimeframeParserPromptInput(message=user_msg))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=TXQAQueryExtraction,
        system_instructions=system_prompt,
    )
    
    return llm_structured_response