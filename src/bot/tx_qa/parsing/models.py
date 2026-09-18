from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel

from bot.tx_qa.domain import Direction, Label
from bot.tx_qa.timeframe.models import Timeframe


@dataclass(frozen=True)
class TxQuery:
    label: Label | None
    start_date: date
    end_date: date
    direction: Direction | None
    
class TxQueryExtraction(BaseModel):
    timeframe: Timeframe
    label: Label | None = None
    direction: Direction | None = None
    # Optional reason or explanation for the parsed timeframe information, can be used for debugging or logging purposes
    reason: str | None = None
    
class TXExplainQuery(BaseModel):
    # Related reference offset - this, current, last = 0; previous = 1
    reference_offset: int | None = None
    # Number of related references (1 = that sum, 2 = previous two sums...)
    reference_count: int | None = None
    
class ExplainTxSummaryQueryExtraction(BaseModel):
    query: TXExplainQuery
    reason: str | None = None
    
