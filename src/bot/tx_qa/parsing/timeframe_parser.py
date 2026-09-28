from bot.llm.client import LLMClient
from bot.tx_qa.parsing.models import TxQueryExtraction
from bot.tx_qa.parsing.timeframe_parser_prompt_loader import (
    TimeframeParserPromptInput,
    TimeframeParserPromptLoader,
)


class TimeframeParser:
    def __init__(self, llm_client: LLMClient, prompt_loader: TimeframeParserPromptLoader):
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader

    def parse(self, user_msg: str) -> TxQueryExtraction:
        system_prompt = self._prompt_loader.load_system_instructions()
        user_prompt = self._prompt_loader.build_user_prompt(
            input=TimeframeParserPromptInput(message=user_msg)
        )

        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=TxQueryExtraction,
            system_instructions=system_prompt,
        )
        return llm_structured_response
