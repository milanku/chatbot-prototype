from bot.tx_qa.memory.models import SessionState, SessionStore


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        return self._sessions.setdefault(session_id, SessionState())

    def set_session(self, session_id: str, session_state: SessionState) -> None:
        self._sessions[session_id] = session_state
