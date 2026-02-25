from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")
    
def log_event(*, trace_id: str, event: str, payload: Dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event,
        "payload": payload,
    }
    logging.getLogger("bot").info(json.dumps(record, ensure_ascii=False))