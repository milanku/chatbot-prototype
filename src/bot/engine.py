from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from bot.logging import log_event
from bot.models.responses import BotResponse


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

        # Mock answer for now
        answer_text = (
            "No proper answer yet.\n"
        )

        resp = BotResponse(answer=answer_text, references=[], trace_id=trace_id)

        # Trace: engine finish
        log_event(
            trace_id=trace_id,
            event="engine.finish",
            payload={"references": resp.references},
        )

        return resp