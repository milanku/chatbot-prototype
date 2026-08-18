from pathlib import Path

from bot.llm.client import LLMClient
from bot.models.tx_qa.query import TXExplainQueryExtraction
from bot.routes.tx_qa.explain_prompt_builder import (
    TXExplainParserPromptInput,
    build_tx_explain_parser_system_prompt,
    build_tx_explain_parser_user_prompt,
    load_tx_explain_parser_instructions,
)


def parse_explain_query_from_user_message(llm_client: LLMClient, msg: str) -> TXExplainQueryExtraction | None:
    prompt_template = load_tx_explain_parser_instructions(Path("src/bot/prompts/explain_parse_instructions.txt"))
    system_prompt = build_tx_explain_parser_system_prompt(
        template=prompt_template
    )
    user_prompt = build_tx_explain_parser_user_prompt(TXExplainParserPromptInput(message=msg))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=TXExplainQueryExtraction,
        system_instructions=system_prompt,
    )
    
    return llm_structured_response