from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(Enum):
    TX_SUMMARY = "TX_SUMMARY" # Request for aggregate information about transactions
    TX_EXPLAIN = "TX_EXPLAIN" # List transactions that make up a previously given summary
    TX_LIST = "TX_LIST" # List transactions matching a new user query without reference to a previous summary
    DOCS_ANSWER = "DOCS_ANSWER" # Answer questions based on documentation
    OUT_OF_SCOPE = "OUT_OF_SCOPE" # Messages that do not fit any of the above routes


@dataclass(frozen=True)
class RouterDecision:
    route: Route
    confidence: float
    reason: str | None = None
