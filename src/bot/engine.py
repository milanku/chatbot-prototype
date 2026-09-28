from __future__ import annotations

from datetime import datetime

from bot.bot_models import BotResponse
from bot.common.lazy import Lazy
from bot.engine_models import EngineResponse
from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.logging import generate_id, log_event
from bot.routing.models import ChatbotRouter, Route, RouterDecision
from bot.trace_context import bind_trace_id
from bot.tx_qa.memory.models import SessionState


class ChatbotEngine:
    def __init__(
        self,
        *,
        route_selector: ChatbotRouter,
        tx_summary_handler: RouteHandler,
        explain_tx_summary_handler: RouteHandler,
        tx_list_handler: RouteHandler,
        docs_answer_handler: Lazy[RouteHandler],
        out_of_scope_handler: RouteHandler,
        unknown_route_handler: RouteHandler,
    ) -> None:
        self._route_selector = route_selector
        self._tx_summary_handler = tx_summary_handler
        self._explain_tx_summary_handler = explain_tx_summary_handler
        self._tx_list_handler = tx_list_handler
        self._docs_answer_handler = docs_answer_handler
        self._out_of_scope_handler = out_of_scope_handler
        self._unknown_route_handler = unknown_route_handler

    def answer(self, message: str, *, session_state: SessionState) -> EngineResponse:
        current_time = datetime.now()
        trace_id = generate_id(current_time)

        with bind_trace_id(trace_id):
            log_event(
                event="engine.start",
                payload={
                    "message": message,
                },
            )

            # Main router logic
            router_decision: RouterDecision = self._route_selector.select(message=message)

            log_event(
                event="router.decision",
                payload={
                    "route": router_decision.route.value,
                },
            )

            match router_decision.route:
                case Route.TX_SUMMARY:
                    result: RouteHandlerResult = self._tx_summary_handler.handle(
                        message=message,
                        session_state=session_state,
                    )
                case Route.TX_LIST:
                    result = self._tx_list_handler.handle(
                        message=message,
                        session_state=session_state,
                    )
                case Route.EXPLAIN_TX_SUMMARY:
                    result = self._explain_tx_summary_handler.handle(
                        message=message,
                        session_state=session_state,
                    )
                case Route.DOCS_ANSWER:
                    result = self._docs_answer_handler.get().handle(
                        message=message,
                        session_state=session_state,
                    )
                case Route.OUT_OF_SCOPE:
                    result = self._out_of_scope_handler.handle(
                        message=message,
                        session_state=session_state,
                    )
                case _:
                    result = self._unknown_route_handler.handle(
                        message=message,
                        session_state=session_state,
                    )

            log_event(
                event="engine.finish",
                payload={
                    "answer": result.answer_text,
                    "doc_references": result.references,
                },
            )

        bot_response = BotResponse(
            answer=result.answer_text,
            doc_references=result.references,
            trace_id=trace_id,
        )

        return EngineResponse(
            response=bot_response,
            new_state=result.new_state,
        )
