from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, date
import dataclasses
import enum
from typing import Any


def setup_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _default_serializer(o: Any) -> Any:
    # dataclasses -> dict
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)

    # datetime/date -> ISO format
    if isinstance(o, (date, datetime)):
        return o.isoformat()

    # Enums -> their value
    if isinstance(o, enum.Enum):
        return o.value

    # Sets -> list
    if isinstance(o, set):
        return list(o)

    # Objects providing to_dict
    if hasattr(o, "to_dict") and callable(getattr(o, "to_dict")):
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


def log_event(*, trace_id: str, event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event,
        "payload": payload,
    }

    try:
        text = json.dumps(record, ensure_ascii=False, default=_default_serializer)
    except TypeError:
        # Very defensive: fall back to a best-effort string representation
        try:
            record["payload"] = str(payload)
            text = json.dumps(record, ensure_ascii=False)
        except Exception:
            text = f"{record['ts']} {trace_id} {event} (unserializable payload)"

    logging.getLogger("bot").debug(text)