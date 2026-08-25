from typing import Protocol

from bot.models.memory import SessionState


class SessionStore(Protocol):
    def get_session(self, session_id: str) -> SessionState:
        ...

    def set_session(self, session_id: str, session_state: SessionState) -> None:
        ...