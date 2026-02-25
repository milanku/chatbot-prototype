from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TRecipe = Literal["TX_SUMMARY", "TX_EXPLAIN", "DOCS_ANSWER", "OUT_OF_SCOPE"]


@dataclass(frozen=True)
class RouterDecision:
    recipe: TRecipe
    confidence: float
