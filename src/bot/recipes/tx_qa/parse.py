import re
from dataclasses import dataclass
from datetime import date
from typing import cast

from bot.models.domain import Label

_LABEL_RE = re.compile(r"\b(food|pets|other)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_RANGE_RE = re.compile(r"\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class TxQAQuery:
    label: Label
    start: date
    end: date


def parse_query(msg: str) -> TxQAQuery | None:
    text = msg.strip().lower()

    label_match = _LABEL_RE.search(text)
    if not label_match:
        return None
    label = cast(Label, label_match.group(1).lower())

    range_match = _RANGE_RE.search(text)
    if range_match:
        start_date_str, end_date_str = range_match.groups()
    else:
        date_matches = _DATE_RE.findall(text)
        if len(date_matches) >= 2:
            start_date_str, end_date_str = date_matches[:2]
        else:
            return None

    start = date.fromisoformat(start_date_str)
    end = date.fromisoformat(end_date_str)

    return TxQAQuery(label=label, start=start, end=end)
