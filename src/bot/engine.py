from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

from langchain_core.embeddings import Embeddings

from bot.composer.handlers.doc_qa import create_docs_answer_handler
from bot.composer.handlers.tx_list import create_tx_list_handler
from bot.config.prompts_config import PromptConfigs
from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.handlers.out_of_scope import OutOfScopeHandler
from bot.handlers.tx_explain import TxExplainHandler
from bot.handlers.tx_summary import TxSummaryHandler
from bot.handlers.unknown_route import UnknownRouteHandler
from bot.llm import client
from bot.logging import log_event
from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState
from bot.models.responses import BotResponse
from bot.models.routing import Route, RouterDecision
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.doc_qa.doc_store import DocStore
from bot.routing.router import select_route
from bot.trace_context import bind_trace_id, get_current_session_id


class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    app_name: str = "Chatbot Prototype"


@dataclass(frozen=True)
class EngineDeps:
    tx_repository: TransactionsRepository
    doc_repository: DocStore
    embedder: Embeddings
    llm_client: client.LLMClient
    prompt_configs: PromptConfigs
    

class ChatbotEngine:
    def __init__(self, config: EngineConfig, deps: EngineDeps) -> None:
        self._config = config
        self._deps = deps
        self._docs = deps.doc_repository
        self._tx_summary_handler: RouteHandler = TxSummaryHandler(
            tx_repository=deps.tx_repository,
            llm_client=deps.llm_client,
            timeframe_parser_prompt_loader=self._deps.prompt_configs.timeframe_parser
        )
        self._tx_list_handler: RouteHandler = create_tx_list_handler(
            llm_client=deps.llm_client,
            tx_repository=deps.tx_repository,
            timeframe_parser_prompt_config=self._deps.prompt_configs.timeframe_parser
        )
        self._tx_explain_handler: RouteHandler = TxExplainHandler(
            llm_client=deps.llm_client,
            explain_query_parser_prompt_loader=self._deps.prompt_configs.explain_parser
        )
        self._docs_answer_handler: RouteHandler = create_docs_answer_handler(
            llm_client=deps.llm_client,
            doc_store=deps.doc_repository,
            embedder=deps.embedder,
            answer_synthesizer_prompt_config=deps.prompt_configs.doc_answer_synthesizer,
            claim_extractor_prompt_config=deps.prompt_configs.claim_extractor,
            claim_verifier_prompt_config=deps.prompt_configs.claim_verifier,
        )
        self._out_of_scope_handler: RouteHandler = OutOfScopeHandler()
        self._unknown_route_handler: RouteHandler = UnknownRouteHandler()
        
    def answer(
        self, message: str, *, session_state: SessionState
    ) -> tuple[BotResponse, SessionState]:
        trace_id = uuid4().hex
        session_id = get_current_session_id()
        
        new_state = session_state  # By default, the state doesn't change. Routes can override this if needed.
        with bind_trace_id(trace_id):
            log_event(
                event="engine.start",
                payload={
                    "session_id": session_id,
                    "message": message,
                }
            )

            router_decision: RouterDecision = select_route(
                llm_client=self._deps.llm_client,
                message=message,
                prompt_loader=self._deps.prompt_configs.router
            )

            log_event(
                event="router.decision",
                payload={
                    "route": router_decision.route.value,
                }
            )
            references: list[DocReference] = []

            match router_decision.route:
                case Route.TX_SUMMARY:
                    result: RouteHandlerResult = self._tx_summary_handler.handle(message=message, session_state=session_state)
                case Route.TX_LIST:
                    result = self._tx_list_handler.handle(message=message, session_state=session_state)
                case Route.TX_EXPLAIN:
                    result = self._tx_explain_handler.handle(message=message, session_state=session_state)
                case Route.DOCS_ANSWER:
                    result = self._docs_answer_handler.handle(message=message, session_state=session_state)
                case Route.OUT_OF_SCOPE:
                    result = self._out_of_scope_handler.handle(message=message, session_state=session_state)
                case _:
                    result = self._unknown_route_handler.handle(message=message, session_state=session_state)

            new_state = result.new_state
            answer_text = result.answer_text
            references = result.references

            log_event(
                event="engine.finish",
                payload={"doc_references": [asdict(ref) for ref in references]},
            )

        bot_response = BotResponse(answer=answer_text, doc_references=references, trace_id=trace_id)

        return bot_response, new_state