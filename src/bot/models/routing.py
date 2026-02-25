from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Recipe(Enum):
    TX_SUMMARY = "TX_SUMMARY"
    TX_EXPLAIN = "TX_EXPLAIN"
    DOCS_ANSWER = "DOCS_ANSWER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class RouterDecision:
    recipe: Recipe
    confidence: float
