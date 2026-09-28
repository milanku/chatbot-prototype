from __future__ import annotations

import dataclasses
import enum
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel
from uuid_utils import uuid4

from bot.trace_context import get_current_session_id, get_current_trace_id


class LogLevel(enum.StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


_json_indent: int | None = None

LOGS_DIR = Path("logs")
_OMITTED_LOG_FIELDS = {"embedding"}


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
    # Pydantic models -> dict
    if isinstance(o, BaseModel):
        return o.model_dump(exclude=_OMITTED_LOG_FIELDS)

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
        return o.to_dict()

    # Fallback to __dict__ when available
    if hasattr(o, "__dict__"):
        return o.__dict__

    # Last resort
    return str(o)


def generate_id(time: datetime) -> str:
    return f"{time.strftime('%Y-%m-%d_%H:%M:%S')}_{uuid4().hex[:4]}"


def get_current_log_path() -> Path:
    session_id = get_current_session_id()
    trace_id = get_current_trace_id()

    if session_id and trace_id:
        return LOGS_DIR / "all" / session_id / f"{trace_id}.log"
    return LOGS_DIR / "default.log"


def get_log_path_by_log_level(log_level: LogLevel) -> Path:
    session_id = get_current_session_id()
    trace_id = get_current_trace_id()

    if session_id and trace_id:
        return (
            LOGS_DIR
            / log_level.value.lower()
            / session_id
            / f"{trace_id}_{log_level.value.lower()}.log"
        )
    return LOGS_DIR / log_level.value.lower() / f"default_{log_level.value.lower()}.log"


def _serialize_record(record: dict[str, Any]) -> str:
    try:
        return json.dumps(
            record,
            ensure_ascii=False,
            default=_default_serializer,
            indent=_json_indent,
        )
    except Exception as exc:
        return json.dumps(
            {
                "ts": record["ts"],
                "session_id": record["session_id"],
                "trace_id": record["trace_id"],
                "event": "logging.serialization_failed",
                "original_event": record["event"],
                "error": repr(exc),
            },
            ensure_ascii=False,
            indent=_json_indent,
        )


def log_event(
    *,
    event: str,
    payload: dict[str, Any],
    log_level: LogLevel = LogLevel.DEBUG,
) -> None:
    path = get_current_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_level_path = get_log_path_by_log_level(log_level)
    log_level_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": get_current_session_id(),
        "log_level": log_level.value,
        "trace_id": get_current_trace_id(),
        "event": event,
        "payload": payload,
    }

    text = _serialize_record(record)

    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    with log_level_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")

    if log_level == LogLevel.INFO:
        logging.getLogger("bot").info("\n\n%s", text)
    elif log_level == LogLevel.WARNING:
        logging.getLogger("bot").warning("\n\n%s", text)
    elif log_level == LogLevel.ERROR:
        logging.getLogger("bot").error("\n\n%s", text)

    logging.getLogger("bot").debug("\n\n%s", text)
