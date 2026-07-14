from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT_TRACE_ID: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def get_current_trace_id() -> str | None:
    return _CURRENT_TRACE_ID.get()


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[None]:
    token = _CURRENT_TRACE_ID.set(trace_id)
    try:
        yield
    finally:
        _CURRENT_TRACE_ID.reset(token)
