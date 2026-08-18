from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

_CURRENT_TRACE_ID: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
_CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar("current_session_id", default=None)


def get_current_trace_id() -> str | None:
    return _CURRENT_TRACE_ID.get()

@contextmanager
def bind_trace_id(trace_id: str) -> Generator[None]:
    token = _CURRENT_TRACE_ID.set(trace_id)
    try:
        yield
    finally:
        _CURRENT_TRACE_ID.reset(token)
        
def get_current_session_id() -> str | None:
    return _CURRENT_SESSION_ID.get()

@contextmanager
def bind_session_id(session_id: str) -> Generator[None]:
    token = _CURRENT_SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _CURRENT_SESSION_ID.reset(token)
