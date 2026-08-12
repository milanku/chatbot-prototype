import json
from dataclasses import asdict, dataclass, replace

import pytest

from bot.models.tx_qa.domain import Direction, Label
from bot.models.tx_qa.query import (
    RawRangeEndpoint,
    RollingRangeMode,
    RollingRangeUnit,
    TimeframeType,
    TXQAParseRawQueryData,
)
from bot.routes.tx_qa.deserialize import deserialize_raw_query_data


@dataclass(frozen=True)
class RawQuery:
    label: str | None
    direction: str | None
    timeframe_type: str | None
    relative_offset: int | None
    year: int | None
    quarter: int | None
    month: int | None
    day: int | None
    start_year: int | None
    start_quarter: int | None
    start_month: int | None
    start_day: int | None
    end_year: int | None
    end_quarter: int | None
    end_month: int | None
    end_day: int | None
    unit: str | None
    mode: str | None
    amount: int | None
    confidence: float | None
    reason: str | None


def create_raw_query_data(**overrides: object) -> RawQuery:
    base = RawQuery(
        label="food",
        direction="spend",
        timeframe_type="RELATIVE_DAY",
        relative_offset=1,
        year=2023,
        quarter=1,
        month=1,
        day=1,
        start_year=2023,
        start_quarter=1,
        start_month=1,
        start_day=1,
        end_year=2023,
        end_quarter=1,
        end_month=1,
        end_day=1,
        unit="day",
        mode="previous_complete",
        amount=2,
        confidence=0.95,
        reason="test_reason"
        
    )
    return replace(base, **overrides)

def serialize_raw_query(raw: RawQuery) -> str:
    raw_query_string = json.dumps(asdict(raw))
    return raw_query_string

def test_deserialize_raw_query_data():
    expected = TXQAParseRawQueryData(
        label=Label.FOOD,
        direction=Direction.SPEND,
        timeframe_type=TimeframeType.RELATIVE_DAY,
        relative_offset=1,
        year=2023,
        quarter=1,
        month=1,
        day=1,
        start_endpoint=RawRangeEndpoint(
            year=2023,
            quarter=1,
            month=1,
            day=1
        ),
        end_endpoint=RawRangeEndpoint(
            year=2023,
            quarter=1,
            month=1,
            day=1
        ),
        unit=RollingRangeUnit.DAY,
        mode=RollingRangeMode.PREVIOUS_COMPLETE,
        amount=2,
        confidence=0.95,
        reason="test_reason"
    )
    
    raw_query_data = create_raw_query_data()
    serialized_raw_query = serialize_raw_query(raw_query_data)
    deserialized_data = deserialize_raw_query_data(serialized_raw_query)
    
    assert deserialized_data == expected
    

### TESTS ###

def test_raise_value_error_on_nonjson():
    with pytest.raises(ValueError, match="The LLM response must be a valid JSON string"):
        deserialize_raw_query_data("not a json")
    
