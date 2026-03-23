from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def setup_logging(*, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def log_event(*, trace_id: str, event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event,
        "payload": payload,
    }
    logging.getLogger("bot").debug(json.dumps(record, ensure_ascii=False))