from bot.common.lazy import Lazy
from bot.composition.doc_qa import create_docs_answer_handler
from bot.composition.explain_tx_summary import create_explain_tx_summary_handler
from bot.composition.tx_list import create_tx_list_handler
from bot.composition.tx_summary import create_tx_summary_handler
from bot.config.models import BotConfig
from bot.config.prompts_config import PromptConfigs
from bot.engine import ChatbotEngine
from bot.handlers.models import RouteHandler
from bot.handlers.out_of_scope import OutOfScopeHandler
from bot.handlers.unknown_route import UnknownRouteHandler
from bot.llm.client import LLMClient
from bot.llm.factory import create_llm
from bot.routing.router import RouteSelector
from bot.routing.router_prompt_loader import RouterPromptLoader
from bot.tx_qa.indexing.transactions_repository import TransactionsRepositoryFromJsonMock


def create_chatbot_engine(
    *,
    config: BotConfig,
    prompt_configs: PromptConfigs,
) -> ChatbotEngine:
    llm_client: LLMClient = create_llm(config.llm)

    tx_repository = TransactionsRepositoryFromJsonMock.from_json_file(
        config.transactions_mock_file_path
    )

    route_selector = RouteSelector(
        llm_client=llm_client,
        prompt_loader=RouterPromptLoader(prompt_config=prompt_configs.router),
    )

    docs_answer_handler = Lazy[RouteHandler](
        lambda: create_docs_answer_handler(
            llm_client=llm_client,
            docs_dir_path=config.docs_dir_path,
            embeddings_dir_path=config.embeddings_dir_path,
            embedder_config=config.embedder,
            chunker_config=config.chunker,
            answer_synthesizer_prompt_config=prompt_configs.doc_answer_synthesizer,
            claim_extractor_prompt_config=prompt_configs.claim_extractor,
            claim_verifier_prompt_config=prompt_configs.claim_verifier,
            chunk_judge_prompt_config=prompt_configs.chunk_judge,
            retriever_configs=config.retrievers,
            reranker_config=config.reranker,
        )
    )

    tx_summary_handler: RouteHandler = create_tx_summary_handler(
        llm_client=llm_client,
        tx_repository=tx_repository,
        timeframe_parser_prompt_config=prompt_configs.timeframe_parser,
    )

    tx_list_handler: RouteHandler = create_tx_list_handler(
        llm_client=llm_client,
        tx_repository=tx_repository,
        timeframe_parser_prompt_config=prompt_configs.timeframe_parser,
    )

    explain_tx_summary_handler: RouteHandler = create_explain_tx_summary_handler(
        llm_client=llm_client,
        explain_summary_parser_prompt_config=prompt_configs.explain_tx_summary_parser,
    )

    out_of_scope_handler: RouteHandler = OutOfScopeHandler()
    unknown_route_handler: RouteHandler = UnknownRouteHandler()

    return ChatbotEngine(
        route_selector=route_selector,
        docs_answer_handler=docs_answer_handler,
        tx_summary_handler=tx_summary_handler,
        tx_list_handler=tx_list_handler,
        explain_tx_summary_handler=explain_tx_summary_handler,
        out_of_scope_handler=out_of_scope_handler,
        unknown_route_handler=unknown_route_handler,
    )
