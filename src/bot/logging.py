from __future__ import annotations

import dataclasses
import enum
import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Protocol, cast, runtime_checkable

from bot.trace_context import get_current_trace_id

_json_indent: int | None = None


@runtime_checkable
class _SupportsToDict(Protocol):
    def to_dict(self) -> Any: ...


def setup_logging(*, verbose: bool = False, pretty_json: bool | None = None) -> None:
    global _json_indent

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    # Default to pretty logs in verbose mode unless explicitly overridden.
    use_pretty_json = verbose if pretty_json is None else pretty_json
    _json_indent = 2 if use_pretty_json else None


def _default_serializer(o: Any) -> Any:
    # dataclasses -> dict
    if dataclasses.is_dataclass(o):
        if isinstance(o, type):
            return o.__name__
        return dataclasses.asdict(o)

    # datetime/date -> ISO format
    if isinstance(o, (date, datetime)):
        return o.isoformat()

    # Enums -> their value
    if isinstance(o, enum.Enum):
        return o.value

    # Sets -> list
    if isinstance(o, set):
        return list(cast(set[object], o))

    # Objects providing to_dict
    if isinstance(o, _SupportsToDict):
        try:
            return o.to_dict()
        except Exception:
            pass

    # Fallback to __dict__ when available
    if hasattr(o, "__dict__"):
        try:
            return o.__dict__
        except Exception:
            pass

    # Last resort
    return str(o)


def log_event(*, event: str, payload: dict[str, Any], trace_id: str | None = None) -> None:
    resolved_trace_id = trace_id or get_current_trace_id() or "missing-trace-id"
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": resolved_trace_id,
        "event": event,
        "payload": payload,
    }

    try:
        text = json.dumps(
            record,
            ensure_ascii=False,
            default=_default_serializer,
            indent=_json_indent,
        )
    except TypeError:
        # Very defensive: fall back to a best-effort string representation
        try:
            record["payload"] = str(payload)
            text = json.dumps(record, ensure_ascii=False, indent=_json_indent)
        except Exception:
            text = f"{record['ts']} {resolved_trace_id} {event} (unserializable payload)"

    logging.getLogger("bot").debug(f"\n\n{text}")