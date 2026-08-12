from dataclasses import replace

import pytest
from typing_extensions import Protocol, TypedDict, Unpack

from bot.models.tx_qa.query import (
    TimeframeType,
    TXQAParseRawQueryData,
)


class RawQueryDataOverrides(TypedDict, total=False):
    """
    Fields that tests are allowed to override.

    Add other TXQAParseRawQueryData fields here when tests need them.
    """

    timeframe_type: TimeframeType
    year: int | None
    month: int | None
    day: int | None


class RawQueryDataFactory(Protocol):
    def __call__(
        self,
        **overrides: Unpack[RawQueryDataOverrides],
    ) -> TXQAParseRawQueryData:
        ...


BASE_RAW_QUERY_DATA = TXQAParseRawQueryData(
    label=None,
    direction=None,
    timeframe_type=TimeframeType.UNKNOWN,
    relative_offset=None,
    year=None,
    quarter=None,
    month=None,
    day=None,
    start_endpoint=None,
    end_endpoint=None,
    unit=None,
    mode=None,
    amount=None,
    confidence=None,
    reason=None,
)


@pytest.fixture
def raw_query_data_factory() -> RawQueryDataFactory:
    """
    Return a function that creates TXQAParseRawQueryData instances.

    BASE_RAW_QUERY_DATA is frozen, so replace() creates a new object
    rather than changing the shared baseline.
    """

    def create(
        **overrides: Unpack[RawQueryDataOverrides],
    ) -> TXQAParseRawQueryData:
        return replace(BASE_RAW_QUERY_DATA, **overrides)

    return create