from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Route(Enum):
    TX_SUMMARY = "TX_SUMMARY" # Request for aggregate information about transactions
    EXPLAIN_TX_SUMMARY = "EXPLAIN_TX_SUMMARY" # List transactions that make up a previously given summary
    TX_LIST = "TX_LIST" # List transactions matching a new user query without reference to a previous summary
    DOCS_ANSWER = "DOCS_ANSWER" # Answer questions based on documentation
    OUT_OF_SCOPE = "OUT_OF_SCOPE" # Messages that do not fit any of the above routes

class RouterDecisionExtraction(BaseModel):
    decision: RouterDecision
    reason: str | None = None

class RouterDecision(BaseModel):
    route: Route