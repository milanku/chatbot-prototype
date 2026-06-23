from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

from bot.handlers.docs_answer import DocsAnswerHandler
from bot.handlers.models import HandlerResult
from bot.handlers.out_of_scope import OutOfScopeHandler
from bot.handlers.tx_explain import TxExplainHandler
from bot.handlers.tx_list import TxListHandler
from bot.handlers.tx_summary import TxSummaryHandler
from bot.handlers.unknown_route import UnknownRouteHandler
from bot.llm import client
from bot.logging import log_event
from bot.models.doc_qa.docs import DocRepository
from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState
from bot.models.responses import BotResponse
from bot.models.routing import Route, RouterDecision
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routing.router import route


@dataclass(frozen=True)
class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    app_name: str = "chatbot-prototype"


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
        
        self._tx_summary_handler = TxSummaryHandler(tx_repository=deps.tx_repository, llm_client=deps.llm_client)
        self._tx_list_handler = TxListHandler(tx_repository=deps.tx_repository, llm_client=deps.llm_client)
        self._tx_explain_handler = TxExplainHandler(tx_repository=deps.tx_repository, llm_client=deps.llm_client)
        self._docs_answer_handler = DocsAnswerHandler(doc_repository=deps.doc_repository, llm_client=deps.llm_client)
        self._out_of_scope_handler = OutOfScopeHandler()
        self._unknown_route_handler = UnknownRouteHandler()
        
    def answer(
        self, message: str, *, session_id: str, session_state: SessionState
    ) -> tuple[BotResponse, SessionState]:
        trace_id = uuid4().hex
        
        new_state = session_state  # By default, the state doesn't change. Routes can override this if needed.

        def trace(event: str, **payload: object) -> None:
             log_event(trace_id=trace_id, event=event, payload=payload)

        # Trace: engine start
        trace(
            "engine.start",
            session_id=session_id,
            message=message,
            app=self._config.app_name,
        )

        router_decision: RouterDecision = route(session_id=session_id, llm_client=self._deps.llm_client, message=message)

        trace(
            "router.decision",
            route=router_decision.route.value,
            confidence=router_decision.confidence,
        )
        references: list[DocReference] = []

        match router_decision.route:
            case Route.TX_SUMMARY:
                result: HandlerResult = self._tx_summary_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)    
            case Route.TX_LIST:
                result: HandlerResult = self._tx_list_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)
            case Route.TX_EXPLAIN:
                result: HandlerResult = self._tx_explain_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)
            case Route.DOCS_ANSWER:
                result: HandlerResult = self._docs_answer_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)
            case Route.OUT_OF_SCOPE:
                result: HandlerResult = self._out_of_scope_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)
            case _:
                result: HandlerResult = self._unknown_route_handler.handle(message=message, session_id=session_id, session_state=session_state, trace=trace)
        
        new_state = result.new_state
        answer_text = result.answer_text
        references = result.references
        
        bot_response = BotResponse(answer=answer_text, doc_references=references, trace_id=trace_id)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"doc_references": [asdict(ref) for ref in bot_response.doc_references]},
        )

        return bot_response, new_state