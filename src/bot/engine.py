from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

from bot.config.prompts_config import PromptConfig
from bot.handlers.base import Handler
from bot.handlers.docs_answer import DocsAnswerHandler
from bot.handlers.models import HandlerResult
from bot.handlers.out_of_scope import OutOfScopeHandler
from bot.handlers.tx_explain import TxExplainHandler
from bot.handlers.tx_list import TxListHandler
from bot.handlers.tx_summary import TxSummaryHandler
from bot.handlers.unknown_route import UnknownRouteHandler
from bot.llm import client
from bot.logging import log_event
from bot.models.doc_qa.doc_repository import DocRepository
from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState
from bot.models.responses import BotResponse
from bot.models.routing import Route, RouterDecision
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.doc_qa.verifier.verifier_prompt_loader import (
    ClaimVerificationPromptLoader,
)
from bot.routing.prompt_builder import RouterPromptLoader
from bot.routing.router import select_route
from bot.trace_context import bind_trace_id, get_current_session_id


class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    
    def __init__(
        self,
        *,
        router_config: PromptConfig,
        claim_extractor_config: PromptConfig,
        claim_verifier_config: PromptConfig,
        explain_parse_config: PromptConfig,
        timeframe_parser_config: PromptConfig,
        doc_answer_synthesizer_config: PromptConfig,
    ) -> None:
        self.router_config = router_config
        self.claim_extractor_config = claim_extractor_config
        self.claim_verifier_config = claim_verifier_config
        self.explain_parse_config = explain_parse_config
        self.timeframe_parser_config = timeframe_parser_config
        self.doc_answer_synthesizer_config = doc_answer_synthesizer_config


@dataclass(frozen=True)
class EngineDeps:
    tx_repository: TransactionsRepository
    doc_repository: DocRepository
    llm_client: client.LLMClient
    

class ChatbotEngine:
    def __init__(self, config: EngineConfig, deps: EngineDeps) -> None:
        self._config = config
        self._deps = deps
        self._docs = deps.doc_repository
        
        self._tx_summary_handler: Handler = TxSummaryHandler(
            tx_repository=deps.tx_repository,
            llm_client=deps.llm_client,
        )
        self._tx_list_handler: Handler = TxListHandler(
            tx_repository=deps.tx_repository,
            llm_client=deps.llm_client,
        )
        self._tx_explain_handler: Handler = TxExplainHandler(
            tx_repository=deps.tx_repository,
            llm_client=deps.llm_client,
        )
        self._docs_answer_handler: Handler = DocsAnswerHandler(
            doc_repository=deps.doc_repository,
            llm_client=deps.llm_client,
            doc_verifier_prompt_loader=ClaimVerificationPromptLoader(
                prompt_config=self._config.claim_verifier_config
            ),
        )
        self._out_of_scope_handler: Handler = OutOfScopeHandler()
        self._unknown_route_handler: Handler = UnknownRouteHandler()
        
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
                prompt_loader=RouterPromptLoader(prompt_config=self._config.explain_parse_config)
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
                    result: HandlerResult = self._tx_summary_handler.handle(message=message, session_state=session_state)
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