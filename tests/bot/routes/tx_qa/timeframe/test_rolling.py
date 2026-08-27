from datetime import date

import pytest

from bot.models.tx_qa.query import DateRange, RollingRangeUnit
from bot.routes.tx_qa.timeframe.rolling import (
    resolve_previous_complete_range,
    resolve_trailing_range,
)


@pytest.mark.parametrize(
    (
        "unit",
        "amount",
        "today",
        "expected",
    ),
    [
        pytest.param(
            RollingRangeUnit.DAY,
            5,
            date(2026, 1, 30),
            DateRange(
                start_date=date(2026, 1, 26),
                end_date=date(2026, 1, 30),
            ),
            id="trailing-day-offset",
        ),
        pytest.param(
            RollingRangeUnit.DAY,
            6,
            date(2026, 1, 3),
            DateRange(
                start_date=date(2025, 12, 29),
                end_date=date(2026, 1, 3),
            ),
            id="trailing-day-offset-previous-year",
        ),
        pytest.param(
            RollingRangeUnit.WEEK,
            2,
            date(2026, 1, 30),
            DateRange(
                start_date=date(2026, 1, 17),
                end_date=date(2026, 1, 30),
            ),
            id="trailing-week-offset",
        ),
        pytest.param(
            RollingRangeUnit.WEEK,
            2,
            date(2026, 1, 3),
            DateRange(
                start_date=date(2025, 12, 21),
                end_date=date(2026, 1, 3),
            ),
            id="trailing-week-offset-previous-year",
        ),        
        pytest.param(
            RollingRangeUnit.MONTH,
            2,
            date(2026, 6, 20),
            DateRange(
                start_date=date(2026, 4, 20),
                end_date=date(2026, 6, 20),
            ),
            id="trailing-month-offset",
        ),
        pytest.param(
            RollingRangeUnit.MONTH,
            2,
            date(2026, 4, 30),
            DateRange(
                start_date=date(2026, 2, 28),
                end_date=date(2026, 4, 30),
            ),
            id="trailing-month-offset-short-month",
        ),
        pytest.param(
            RollingRangeUnit.YEAR,
            2,
            date(2026, 4, 30),
            DateRange(
                start_date=date(2024, 4, 30),
                end_date=date(2026, 4, 30),
            ),
            id="trailing-year-offset",
        ),
        pytest.param(
            RollingRangeUnit.YEAR,
            2,
            date(2028, 2, 29),
            DateRange(
                start_date=date(2026, 2, 28),
                end_date=date(2028, 2, 29),
            ),
            id="trailing-year-offset-leap-year",
        ),
        
        
    ],
)
def test_range_from_trailing_rolling_range(
    unit: RollingRangeUnit,
    amount: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_trailing_range(
        unit=unit,
        unit_amount=amount,
        today=today
    )

    assert result == expected
    
    
@pytest.mark.parametrize(
    (
        "unit",
        "amount",
        "today",
        "expected",
    ),
    [
        pytest.param(
            RollingRangeUnit.DAY,
            5,
            date(2026, 1, 30),
            DateRange(
                start_date=date(2026, 1, 25),
                end_date=date(2026, 1, 29),
            ),
            id="trailing-day-offset",
        ),
        pytest.param(
            RollingRangeUnit.DAY,
            6,
            date(2026, 1, 3),
            DateRange(
                start_date=date(2025, 12, 28),
                end_date=date(2026, 1, 2),
            ),
            id="trailing-day-offset-previous-year",
        ),
        pytest.param(
            RollingRangeUnit.WEEK,
            2,
            date(2026, 1, 30),
            DateRange(
                start_date=date(2026, 1, 12),
                end_date=date(2026, 1, 25),
            ),
            id="trailing-week-offset",
        ),
        pytest.param(
            RollingRangeUnit.WEEK,
            2,
            date(2026, 1, 3),
            DateRange(
                start_date=date(2025, 12, 15),
                end_date=date(2025, 12, 28),
            ),
            id="trailing-week-offset-previous-year",
        ),        
        pytest.param(
            RollingRangeUnit.MONTH,
            2,
            date(2026, 6, 20),
            DateRange(
                start_date=date(2026, 4, 1),
                end_date=date(2026, 5, 31),
            ),
            id="trailing-month-offset",
        ),
        pytest.param(
            RollingRangeUnit.MONTH,
            2,
            date(2026, 3, 30),
            DateRange(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 28),
            ),
            id="trailing-month-offset-short-month",
        ),
        pytest.param(
            RollingRangeUnit.QUARTER,
            2,
            date(2026, 7, 20),
            DateRange(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 6, 30),
            ),
            id="trailing-quarter-offset",
        ),
        pytest.param(
            RollingRangeUnit.QUARTER,
            2,
            date(2026, 3, 30),
            DateRange(
                start_date=date(2025, 7, 1),
                end_date=date(2025, 12, 31),
            ),
            id="trailing-quarter-offset-previous-year",
        ),
        pytest.param(
            RollingRangeUnit.YEAR,
            2,
            date(2026, 4, 30),
            DateRange(
                start_date=date(2024, 1, 1),
                end_date=date(2025, 12, 31),
            ),
            id="trailing-year-offset",
        ),
        pytest.param(
            RollingRangeUnit.YEAR,
            2,
            date(2028, 2, 29),
            DateRange(
                start_date=date(2026, 1, 1),
                end_date=date(2027, 12, 31),
            ),
            id="trailing-year-offset-leap-year",
        ),
        
        
    ],
)
def test_range_from_previous_complete_rolling_range(
    unit: RollingRangeUnit,
    amount: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_previous_complete_range(
        unit=unit,
        unit_amount=amount,
        today=today
    )

    assert result == expected