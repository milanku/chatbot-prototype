from bot.config.prompts_config import PromptConfig
from bot.handlers.tx_summary import TxSummaryHandler
from bot.llm.client import LLMClient
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.tx_qa.timeframe_parser import TimeframeParser
from bot.routes.tx_qa.timeframe_parser_prompt_loader import TimeframeParserPromptLoader
from bot.routes.tx_qa.tx_summary_coordinator import TxSummaryCoordinator


def create_tx_summary_handler(
    *,
    llm_client: LLMClient,
    tx_repository: TransactionsRepository,
    timeframe_parser_prompt_config: PromptConfig,
) -> TxSummaryHandler:
    
    timeframe_parser = TimeframeParser(
        llm_client=llm_client,
        prompt_loader=TimeframeParserPromptLoader(prompt_config=timeframe_parser_prompt_config),
    )
    
    coordinator = TxSummaryCoordinator(
        tx_repository=tx_repository,
        timeframe_parser=timeframe_parser
    )
    
    return TxSummaryHandler(coordinator=coordinator)