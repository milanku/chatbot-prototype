from bot.llm.client import LLMClient
from bot.models.tx_qa.query import TXExplainQueryExtraction
from bot.routes.tx_qa.explain_parser_prompt_loader import (
    TXExplainParserPromptInput,
    TXExplainParserPromptLoader,
)


def parse_explain_query_from_user_message(
    *,
    llm_client: LLMClient,
    msg: str,
    prompt_loader: TXExplainParserPromptLoader
)-> TXExplainQueryExtraction | None:
    system_prompt = prompt_loader.load_system_instructions();
    user_prompt = prompt_loader.build_user_prompt(TXExplainParserPromptInput(message=msg))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=TXExplainQueryExtraction,
        system_instructions=system_prompt,
    )
    
    return llm_structured_response