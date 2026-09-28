from bot.config.prompts_config import PromptConfig
from bot.handlers.explain_tx_summary import ExplainTxSummaryHandler
from bot.llm.client import LLMClient
from bot.tx_qa.parsing.explain_summary_parser import ExplainTxSummaryParser
from bot.tx_qa.parsing.explain_summary_parser_prompt_loader import (
    ExplainTxSummaryParserPromptLoader,
)
from bot.tx_qa.tx_summary.explain_tx_summary_coordinator import (
    ExplainTxSummaryCoordinator,
)


def create_explain_tx_summary_handler(
    *,
    llm_client: LLMClient,
    explain_summary_parser_prompt_config: PromptConfig,
) -> ExplainTxSummaryHandler:

    explain_summary_parser = ExplainTxSummaryParser(
        llm_client=llm_client,
        prompt_loader=ExplainTxSummaryParserPromptLoader(
            prompt_config=explain_summary_parser_prompt_config
        ),
    )

    coordinator = ExplainTxSummaryCoordinator(explain_summary_parser=explain_summary_parser)

    return ExplainTxSummaryHandler(coordinator=coordinator)
