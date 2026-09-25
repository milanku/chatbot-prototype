from __future__ import annotations

import dataclasses
import enum
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from uuid_utils import uuid4

from bot.trace_context import get_current_session_id, get_current_trace_id

_json_indent: int | None = None

LOGS_DIR = Path("logs")


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


def generate_id(time: datetime) -> str:
    return f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{uuid4().hex[:4]}"


def get_current_log_path() -> Path:
    session_id = get_current_session_id()
    trace_id = get_current_trace_id()

    if session_id and trace_id:
        return LOGS_DIR / session_id / f"{trace_id}.log"
    return LOGS_DIR / "default.log"


def log_event(*, event: str, payload: dict[str, Any]) -> None:
    path = get_current_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": get_current_session_id(),
        "trace_id": get_current_trace_id(),
        "event": event,
        "payload": payload,
    }
    text = ""

    try:
        record_dump = json.dumps(
            record,
            ensure_ascii=False,
            default=_default_serializer,
            indent=_json_indent,
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(record_dump + "\n")
    except TypeError:
        # Fallback to a best-effort string representation
        try:
            record["payload"] = str(payload)
            text = json.dumps(record, ensure_ascii=False, indent=_json_indent)
        except Exception:
            text = f"{record['ts']} {get_current_trace_id()} {event} (unserializable payload)"

    logging.getLogger("bot").debug(f"\n\n{record if 'record' in locals() else text}")
