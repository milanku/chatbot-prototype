from dataclasses import dataclass, field

from bot.models.memory import SessionState


@dataclass
class InMemorySessionStore:
    _sessions: dict[str, SessionState] = field(default_factory=dict)

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def set_session(self, session_id: str, session_state: SessionState) -> None:
        self._sessions[session_id] = session_state
