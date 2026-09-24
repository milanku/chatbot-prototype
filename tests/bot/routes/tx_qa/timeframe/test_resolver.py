from datetime import date
from unittest.mock import call, patch

import pytest

from bot.tx_qa.timeframe.models import (
    RawRangeEndpoint,
    RollingRangeMode,
    RollingRangeUnit,
    Timeframe,
    TimeframeType,
)
from bot.tx_qa.timeframe.resolve import resolve_date_range

TODAY = date(2025, 1, 3)
SENTINEL_RANGE = (
    date(2001, 2, 3),
    date(2004, 5, 6),
)
RAW_RANGE_ENDPOINT_2025_05_03 = RawRangeEndpoint(
    year=2025,
    quarter=None,
    month=5,
    day=3,
)
RAW_RANGE_ENDPOINT_2025_05_10 = RawRangeEndpoint(
    year=2025,
    quarter=None,
    month=5,
    day=10,
)

def create_timeframe_query(**overrides: object) -> Timeframe:
    return Timeframe(
        timeframe_type=TimeframeType.UNKNOWN,
    ).model_copy(update=overrides)


@pytest.mark.parametrize(
    ("timeframe_type", "raw_overrides", "resolver_name", "expected_call"),
    [
        pytest.param(
            TimeframeType.NAMED_YEAR,
            {"year": 2025},
            "resolve_named_year",
            call(year=2025),
            id="named-year",
        ),
        pytest.param(
            TimeframeType.NAMED_MONTH,
            {"month": 5, "year": 2025},
            "resolve_named_month",
            call(year=2025, month=5, today=TODAY),
            id="named-month",
        ),
        pytest.param(
            TimeframeType.NAMED_MONTH,
            {"month": 5},
            "resolve_named_month",
            call(year=None, month=5, today=TODAY),
            id="named-month-without-year",
        ),
        pytest.param(
            TimeframeType.NAMED_QUARTER,
            {"quarter": 2, "year": 2025},
            "resolve_named_quarter",
            call(year=2025, quarter=2, today=TODAY),
            id="named-quarter",
        ),
        pytest.param(
            TimeframeType.NAMED_QUARTER,
            {"quarter": 2},
            "resolve_named_quarter",
            call(year=None, quarter=2, today=TODAY),
            id="named-quarter-without-year",
        ),
        pytest.param(
            TimeframeType.NAMED_WEEKDAY,
            {"day": 3, "relative_offset": -1},
            "resolve_named_weekday",
            call(weekday=3, offset=-1, today=TODAY),
            id="named-weekday",
        ),
        pytest.param(
            TimeframeType.NAMED_WEEKDAY,
            {"day": 3},
            "resolve_named_weekday",
            call(weekday=3, offset=None, today=TODAY),
            id="named-weekday-without-offset",
        ),
        pytest.param(
            TimeframeType.NAMED_DATE,
            {"year": 2025, "month": 5, "day": 3},
            "resolve_named_date",
            call(year=2025, month=5, day=3, today=TODAY),
            id="named-date",
        ),
        pytest.param(
            TimeframeType.NAMED_DATE,
            {"day": 20},
            "resolve_named_date",
            call(
                year=None,
                month=None,
                day=20,
                today=TODAY,
            ),
            id="named-date-day-only",
        ),
        pytest.param(
            TimeframeType.RELATIVE_YEAR,
            {"relative_offset": -1},
            "resolve_relative_year",
            call(offset=-1, today=TODAY),
            id="relative-year",
        ),
        pytest.param(
            TimeframeType.RELATIVE_MONTH,
            {"relative_offset": 0},
            "resolve_relative_month",
            call(offset=0, today=TODAY),
            id="relative-month",
        ),
        pytest.param(
            TimeframeType.RELATIVE_WEEK,
            {"relative_offset": -1},
            "resolve_relative_week",
            call(offset=-1, today=TODAY),
            id="relative-week",
        ),
        pytest.param(
            TimeframeType.RELATIVE_DAY,
            {"relative_offset": 0},
            "resolve_relative_day",
            call(offset=0, today=TODAY),
            id="relative-day",
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {
                "start_endpoint": RAW_RANGE_ENDPOINT_2025_05_03,
                "end_endpoint": RAW_RANGE_ENDPOINT_2025_05_10,
            },
            "resolve_explicit_range",
            call(
                start_endpoint=RAW_RANGE_ENDPOINT_2025_05_03,
                end_endpoint=RAW_RANGE_ENDPOINT_2025_05_10,
                today=TODAY,
            ),
            id="explicit-range",
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.TRAILING, "unit": RollingRangeUnit.DAY, "unit_amount": 5},
            "resolve_trailing_range",
            call(unit=RollingRangeUnit.DAY, unit_amount=5, today=TODAY),
            id="rolling-range",
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.PREVIOUS_COMPLETE, "unit": RollingRangeUnit.WEEK, "unit_amount": 2},
            "resolve_previous_complete_range",
            call(unit=RollingRangeUnit.WEEK, unit_amount=2, today=TODAY),
            id="rolling-range-previous-complete",
        )
    ],
)

def test_dispatches_to_correct_resolver_with_expected_arguments(
    timeframe_type: TimeframeType,
    raw_overrides: dict[str, object],
    resolver_name: str,
    expected_call: object,
) -> None:

    raw = create_timeframe_query(
        timeframe_type=timeframe_type,
        **raw_overrides
    )

    with patch(
        f"bot.tx_qa.timeframe.resolve.{resolver_name}",
        autospec=True,
        return_value=SENTINEL_RANGE,
    ) as resolver:
        result = resolve_date_range(
            raw,
            today=TODAY,
        )

    assert result == SENTINEL_RANGE
    resolver.assert_called_once()
    assert resolver.call_args == expected_call
    

@pytest.mark.parametrize(
    ("timeframe_type", "raw_overrides", "resolver_name"),
    [
        pytest.param(
            TimeframeType.NAMED_YEAR,
            {},
            "resolve_named_year",
            id="named-year"
        ),
        pytest.param(
            TimeframeType.NAMED_MONTH,
            {},
            "resolve_named_month",
            id="named-month"
        ),
        pytest.param(
            TimeframeType.NAMED_QUARTER,
            {},
            "resolve_named_quarter",
            id="named-quarter"
        ),
        pytest.param(
            TimeframeType.NAMED_WEEKDAY,
            {},
            "resolve_named_weekday",
            id="named-weekday"
        ),
        pytest.param(
            TimeframeType.NAMED_DATE,
            {},
            "resolve_named_date",
            id="named-date"
        ),
        pytest.param(
            TimeframeType.RELATIVE_YEAR,
            {},
            "resolve_relative_year",
            id="relative-year"
        ),
        pytest.param(
            TimeframeType.RELATIVE_MONTH,
            {},
            "resolve_relative_month",
            id="relative-month"
        ),
        pytest.param(
            TimeframeType.RELATIVE_WEEK,
            {},
            "resolve_relative_week",
            id="relative-week"
        ),
        pytest.param(
            TimeframeType.RELATIVE_DAY,
            {},
            "resolve_relative_day",
            id="relative-day"
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {},
            "resolve_explicit_range",
            id="explicit-range"
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {
                "end_endpoint": RawRangeEndpoint(day=10),
            },
            "resolve_explicit_range",
            id="explicit-range-missing-start",
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {
                "start_endpoint": RawRangeEndpoint(),
                "end_endpoint": RawRangeEndpoint(day=10),
            },
            "resolve_explicit_range",
            id="explicit-range-empty-start",
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {
                "start_endpoint": RawRangeEndpoint(month=5)
            },
            "resolve_explicit_range",
            id="explicit-range-missing-end",
        ),
        pytest.param(
            TimeframeType.EXPLICIT_RANGE,
            {
                "start_endpoint": RawRangeEndpoint(day=5),
                "end_endpoint": RawRangeEndpoint(),
            },
            "resolve_explicit_range",
            id="explicit-range-empty-end",
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.TRAILING},
            "resolve_trailing_range",
            id="rolling-range"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.TRAILING, "unit": RollingRangeUnit.DAY},
            "resolve_trailing_range",
            id="rolling-range-missing-amount"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.TRAILING, "unit_amount": 5},
            "resolve_trailing_range",
            id="rolling-range-missing-unit"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.PREVIOUS_COMPLETE},
            "resolve_previous_complete_range",
            id="rolling-range-previous-complete"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.PREVIOUS_COMPLETE, "unit": RollingRangeUnit.DAY},
            "resolve_previous_complete_range",
            id="rolling-range-previous-complete-missing-amount"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": RollingRangeMode.PREVIOUS_COMPLETE, "unit_amount": 5},
            "resolve_previous_complete_range",
            id="rolling-range-previous-complete-missing-unit"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": None, "unit_amount": 5, "unit": RollingRangeUnit.DAY},
            "resolve_trailing_range",
            id="rolling-missing-mode-does-not-call-trailing"
        ),
        pytest.param(
            TimeframeType.ROLLING_RANGE,
            {"mode": None, "unit_amount": 5, "unit": RollingRangeUnit.DAY},
            "resolve_previous_complete_range",
            id="rolling-missing-mode-does-not-call-previous-complete"
        ),
    ]
)
def test_does_not_dispatch_when_required_data_is_missing(
    timeframe_type: TimeframeType,
    raw_overrides: dict[str, object],
    resolver_name: str,
) -> None:
    raw = create_timeframe_query(
        timeframe_type=timeframe_type,
        **raw_overrides,
    )

    with (
        patch(
            f"bot.tx_qa.timeframe.resolve.{resolver_name}",
            autospec=True,
        ) as resolver,
    ):
        result = resolve_date_range(
            raw,
            today=TODAY,
        )

    assert result is None
    resolver.assert_not_called()
    
def test_returns_none_for_unknown_timeframe() -> None:
    raw = create_timeframe_query(
        timeframe_type=TimeframeType.UNKNOWN,
    )

    result = resolve_date_range(
        raw,
        today=TODAY,
    )

    assert result is None
    
def test_propagates_none_from_resolver() -> None:
    raw = create_timeframe_query(
        timeframe_type=TimeframeType.NAMED_MONTH,
        month=5,
    )

    with (
        patch(
            "bot.tx_qa.timeframe.resolve.resolve_named_month",
            autospec=True,
            return_value=None,
        ) as resolver,
    ):
        result = resolve_date_range(
            raw,
            today=TODAY,
        )

    assert result is None

    resolver.assert_called_once_with(
        year=None,
        month=5,
        today=TODAY,
    )