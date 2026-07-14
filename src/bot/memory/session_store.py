from dataclasses import dataclass, field

from bot.models.memory import SessionState


@dataclass
class InMemorySessionStore:
    _sessions: dict[str, SessionState] = field(default_factory=lambda: {})

    def get_session(self, session_id: str) -> SessionState:
        return self._sessions.setdefault(session_id, SessionState())

    def set_session(self, session_id: str, session_state: SessionState) -> None:
        self._sessions[session_id] = session_state
