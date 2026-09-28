from bot.llm.client import LLMClient
from bot.tx_qa.parsing.explain_summary_parser_prompt_loader import (
    ExplainTxSummaryParserPromptInput,
    ExplainTxSummaryParserPromptLoader,
)
from bot.tx_qa.parsing.models import ExplainTxSummaryQueryExtraction


class ExplainTxSummaryParser:
    def __init__(self, *, llm_client: LLMClient, prompt_loader: ExplainTxSummaryParserPromptLoader):
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader

    def parse(self, *, msg: str) -> ExplainTxSummaryQueryExtraction:
        system_prompt = self._prompt_loader.load_system_instructions()
        user_prompt = self._prompt_loader.build_user_prompt(
            ExplainTxSummaryParserPromptInput(message=msg)
        )

        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=ExplainTxSummaryQueryExtraction,
            system_instructions=system_prompt,
        )

        return llm_structured_response
