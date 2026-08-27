from bot.llm.client import LLMClient
from bot.models.tx_qa.query import TXExplainSummaryQueryExtraction
from bot.routes.tx_qa.explain_summary_parser_prompt_loader import (
    TXExplainSummaryParserPromptInput,
    TXExplainSummaryParserPromptLoader,
)


class TXExplainSummaryParser:
    def __init__(self, *, llm_client: LLMClient, prompt_loader: TXExplainSummaryParserPromptLoader):
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader

    def parse(self, *, msg: str) -> TXExplainSummaryQueryExtraction:
        system_prompt = self._prompt_loader.load_system_instructions()
        user_prompt = self._prompt_loader.build_user_prompt(TXExplainSummaryParserPromptInput(message=msg))

        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=TXExplainSummaryQueryExtraction,
            system_instructions=system_prompt,
        )

        return llm_structured_response
     