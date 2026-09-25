from bot.config.prompts_config import PromptConfig
from bot.handlers.tx_list import TxListHandler
from bot.llm.client import LLMClient
from bot.tx_qa.indexing.models import TransactionsRepository
from bot.tx_qa.parsing.timeframe_parser import TimeframeParser
from bot.tx_qa.parsing.timeframe_parser_prompt_loader import TimeframeParserPromptLoader
from bot.tx_qa.tx_list.coordinator import TxListCoordinator


def create_tx_list_handler(
    *,
    llm_client: LLMClient,
    tx_repository: TransactionsRepository,
    timeframe_parser_prompt_config: PromptConfig,
) -> TxListHandler:

    timeframe_parser = TimeframeParser(
        llm_client=llm_client,
        prompt_loader=TimeframeParserPromptLoader(prompt_config=timeframe_parser_prompt_config),
    )

    coordinator = TxListCoordinator(tx_repository=tx_repository, timeframe_parser=timeframe_parser)

    return TxListHandler(coordinator=coordinator)
