from bot.llm.client import LLMClient
from bot.models.tx_qa.query import (
    TXQAQueryExtraction,
)
from bot.routes.tx_qa.timeframe_parser_prompt_loader import (
    TimeframeParserPromptInput,
    TimeframeParserPromptLoader,
)


def parse_raw_tx_query_from_user_message(
    *,
    llm_client: LLMClient,
    user_msg: str,
    prompt_loader: TimeframeParserPromptLoader
) -> TXQAQueryExtraction:
    system_prompt = prompt_loader.load_system_instructions()
    user_prompt = prompt_loader.build_user_prompt(input=TimeframeParserPromptInput(message=user_msg))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=TXQAQueryExtraction,
        system_instructions=system_prompt,
    )
    
    return llm_structured_response