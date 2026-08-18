from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

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
from bot.models.doc_qa.docs import DocRepository
from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState
from bot.models.responses import BotResponse
from bot.models.routing import Route, RouterDecision
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routing.router import route
from bot.trace_context import bind_trace_id


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
        )
        self._out_of_scope_handler: Handler = OutOfScopeHandler()
        self._unknown_route_handler: Handler = UnknownRouteHandler()
        
    def answer(
        self, message: str, *, session_id: str, session_state: SessionState
    ) -> tuple[BotResponse, SessionState]:
        trace_id = uuid4().hex
        
        new_state = session_state  # By default, the state doesn't change. Routes can override this if needed.
        with bind_trace_id(trace_id):
            log_event(
                event="engine.start",
                payload={
                    "session_id": session_id,
                    "message": message,
                    "app": self._config.app_name,
                }
            )

            router_decision: RouterDecision = route(llm_client=self._deps.llm_client, message=message)

            log_event(
                event="router.decision",
                payload={
                    "route": router_decision.route.value,
                }
            )
            references: list[DocReference] = []

            match router_decision.route:
                case Route.TX_SUMMARY:
                    result: HandlerResult = self._tx_summary_handler.handle(message=message, session_id=session_id, session_state=session_state)
                case Route.TX_LIST:
                    result = self._tx_list_handler.handle(message=message, session_id=session_id, session_state=session_state)
                case Route.TX_EXPLAIN:
                    result = self._tx_explain_handler.handle(message=message, session_id=session_id, session_state=session_state)
                case Route.DOCS_ANSWER:
                    result = self._docs_answer_handler.handle(message=message, session_id=session_id, session_state=session_state)
                case Route.OUT_OF_SCOPE:
                    result = self._out_of_scope_handler.handle(message=message, session_id=session_id, session_state=session_state)
                case _:
                    result = self._unknown_route_handler.handle(message=message, session_id=session_id, session_state=session_state)

            new_state = result.new_state
            answer_text = result.answer_text
            references = result.references

            log_event(
                event="engine.finish",
                payload={"doc_references": [asdict(ref) for ref in references]},
            )

        bot_response = BotResponse(answer=answer_text, doc_references=references, trace_id=trace_id)

        return bot_response, new_state