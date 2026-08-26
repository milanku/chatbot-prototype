from bot.config.prompts_config import PromptConfig
from bot.handlers.tx_explain_summary import TxExplainSummaryHandler
from bot.llm.client import LLMClient
from bot.routes.tx_qa.explain_summary_parser import TXExplainSummaryParser
from bot.routes.tx_qa.explain_summary_parser_prompt_loader import (
    TXExplainSummaryParserPromptLoader,
)
from bot.routes.tx_qa.tx_explain_summary_coordinator import TxExplainSummaryCoordinator


def create_tx_explain_summary_handler(
    *,
    llm_client: LLMClient,
    explain_summary_parser_prompt_config: PromptConfig,
 ) -> TxExplainSummaryHandler:
    
    explain_summary_parser = TXExplainSummaryParser(
        llm_client=llm_client,
        prompt_loader=TXExplainSummaryParserPromptLoader(prompt_config=explain_summary_parser_prompt_config),
    )
    
    coordinator = TxExplainSummaryCoordinator(
        explain_summary_parser=explain_summary_parser
    )
    
    return TxExplainSummaryHandler(coordinator=coordinator)