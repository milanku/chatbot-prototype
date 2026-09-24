from __future__ import annotations

from datetime import datetime

from bot.bot_models import BotResponse
from bot.composition.doc_qa import create_docs_answer_handler
from bot.composition.explain_tx_summary import create_explain_tx_summary_handler
from bot.composition.tx_list import create_tx_list_handler
from bot.composition.tx_summary import create_tx_summary_handler
from bot.engine_models import EngineDeps, EngineResponse
from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.handlers.out_of_scope import OutOfScopeHandler
from bot.handlers.unknown_route import UnknownRouteHandler
from bot.logging import generate_id, log_event
from bot.routing.models import Route, RouterDecision
from bot.routing.router import RouteSelector
from bot.routing.router_prompt_loader import RouterPromptLoader
from bot.trace_context import bind_trace_id
from bot.tx_qa.memory.models import SessionState


class ChatbotEngine:
    def __init__(self, deps: EngineDeps) -> None:
        self._embeddings_store = deps.embeddings_store
        self._tx_summary_handler: RouteHandler = create_tx_summary_handler(
            llm_client=deps.llm_client,
            tx_repository=deps.tx_repository,
            timeframe_parser_prompt_config=deps.prompt_configs.timeframe_parser,
        )
        self._tx_list_handler: RouteHandler = create_tx_list_handler(
            llm_client=deps.llm_client,
            tx_repository=deps.tx_repository,
            timeframe_parser_prompt_config=deps.prompt_configs.timeframe_parser
        )
        self._explain_tx_summary_handler: RouteHandler = create_explain_tx_summary_handler(
            llm_client=deps.llm_client,
            explain_summary_parser_prompt_config=deps.prompt_configs.explain_tx_summary_parser
        )
        self._docs_answer_handler: RouteHandler = create_docs_answer_handler(
            llm_client=deps.llm_client,
            embedded_doc_chunks=deps.embeddings_store.get_embedded_chunks(),
            embedder=deps.embedder,
            answer_synthesizer_prompt_config=deps.prompt_configs.doc_answer_synthesizer,
            claim_extractor_prompt_config=deps.prompt_configs.claim_extractor,
            claim_verifier_prompt_config=deps.prompt_configs.claim_verifier,
            chunk_judge_prompt_config=deps.prompt_configs.chunk_judge,
            retriever_configs=deps.retriever_configs,
            reranker_config=deps.reranker_config,
        )
        self._out_of_scope_handler: RouteHandler = OutOfScopeHandler()
        self._unknown_route_handler: RouteHandler = UnknownRouteHandler()
        
        self._route_selector = RouteSelector(
            llm_client=deps.llm_client,
            prompt_loader=RouterPromptLoader(prompt_config=deps.prompt_configs.router)
        )
        
    def answer(
        self,
        message: str,
        *,
        session_state: SessionState
    ) -> EngineResponse:
        current_time = datetime.now()
        trace_id = generate_id(current_time)
        
        with bind_trace_id(trace_id):
            log_event(
                event="engine.start",
                payload={
                    "message": message,
                }
            )

            # Main router logic
            router_decision: RouterDecision = self._route_selector.select(
                message=message
            )

            log_event(
                event="router.decision",
                payload={
                    "route": router_decision.route.value,
                }
            )

            match router_decision.route:
                case Route.TX_SUMMARY:
                    result: RouteHandlerResult = self._tx_summary_handler.handle(message=message, session_state=session_state)
                case Route.TX_LIST:
                    result = self._tx_list_handler.handle(message=message, session_state=session_state)
                case Route.EXPLAIN_TX_SUMMARY:
                    result = self._explain_tx_summary_handler.handle(message=message, session_state=session_state)
                case Route.DOCS_ANSWER:
                    result = self._docs_answer_handler.handle(message=message, session_state=session_state)
                case Route.OUT_OF_SCOPE:
                    result = self._out_of_scope_handler.handle(message=message, session_state=session_state)
                case _:
                    result = self._unknown_route_handler.handle(message=message, session_state=session_state)

            log_event(
                event="engine.finish",
                payload={
                    "answer": result.answer_text,
                    "doc_references": result.references,
                },
            )

        bot_response = BotResponse(answer=result.answer_text, doc_references=result.references, trace_id=trace_id)

        return EngineResponse(response=bot_response, new_state=result.new_state)
