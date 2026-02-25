from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from bot.logging import log_event
from bot.models.responses import BotResponse
from bot.models.routing import Recipe, RouterDecision
from bot.routing.router import route


@dataclass(frozen=True)
class EngineConfig:
    # TODO add: docs path, model names, retrieval parameters, etc.
    app_name: str = "chatbot-prototype"


class ChatbotEngine:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        
    def answer(self, message: str, *, session_id: str = "default") -> BotResponse:
        trace_id = uuid4().hex

        # Trace: engine start
        log_event(
            trace_id=trace_id,
            event="engine.start",
            payload={"session_id": session_id, "message": message, "app": self._config.app_name},
        )

        router_decision: RouterDecision = route(message)

        log_event(
            trace_id=trace_id,
            event="router.decision",
            payload={"recipe": router_decision.recipe.value, "confidence": router_decision.confidence},
        )

        match router_decision.recipe:
            case Recipe.TX_SUMMARY:
                answer_text = "TX_SUMMARY is not implemented yet.\n"
            case Recipe.TX_EXPLAIN:
                answer_text = "TX_EXPLAIN is not implemented yet.\n"
            case Recipe.DOCS_ANSWER:
                answer_text = "DOCS_ANSWER is not implemented yet.\n"
            case Recipe.OUT_OF_SCOPE:
                answer_text = "Sorry, I can't help with that.\n"
            case _:
                answer_text = "Unknown recipe.\n"

        resp = BotResponse(answer=answer_text, references=[], trace_id=trace_id)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"references": resp.references},
        )

        return resp
        return resp
