from datetime import date

import pytest

from bot.models.tx_qa.query import DateRange
from bot.tx_qa.timeframe.calendar_utils import (
    day_range,
    month_range,
    quarter_range,
    year_range,
)


@pytest.mark.parametrize(
    (
        "year",
        "expected",
    ),
    [
        pytest.param(
            2025,
            DateRange(
                start_date=date(2025, 1, 1),
                end_date=date(2025, 12, 31),
            ),
            id="year-range-2025",
        ),
    ],
)
def test_year_range(
    year: int,
    expected: DateRange | None,
) -> None:
    result = year_range(
        year=year
    )

    assert result == expected
    
@pytest.mark.parametrize(
    (
        "year",
        "quarter",
        "expected",
    ),
    [
        pytest.param(
            2025,
            2,
            DateRange(
                start_date=date(2025, 4, 1),
                end_date=date(2025, 6, 30),
            ),
            id="quarter-2-2025",
        ),
    ],
)
def test_quarter_range(
    year: int,
    quarter: int,
    expected: DateRange | None,
) -> None:
    result = quarter_range(
        year=year,
        quarter=quarter,
    )

    assert result == expected
    
@pytest.mark.parametrize(
    (
        "year",
        "month",
        "expected",
    ),
    [
        pytest.param(
            2025,
            4,
            DateRange(
                start_date=date(2025, 4, 1),
                end_date=date(2025, 4, 30),
            ),
            id="month-4-2025",
        ),
        pytest.param(
            2024,
            2,
            DateRange(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 29),
            ),
            id="month-leap-year-2-2024",
        ),
    ],
)
def test_month_range(
    year: int,
    month: int,
    expected: DateRange | None,
) -> None:
    result = month_range(
        year=year,
        month=month,
    )

    assert result == expected
    
@pytest.mark.parametrize(
    (
        "year",
        "month",
        "day",  
        "expected",
    ),
    [
        pytest.param(
            2025,
            4,
            15,
            DateRange(
                start_date=date(2025, 4, 15),
                end_date=date(2025, 4, 15),
            ),
            id="day-15-4-2025",
        ),
        pytest.param(
            2024,
            2,
            29,
            DateRange(
                start_date=date(2024, 2, 29),
                end_date=date(2024, 2, 29),
            ),
            id="day-29-2-2024",
        ),
    ],
)
def test_day_range(
    year: int,
    month: int,
    day: int,
    expected: DateRange | None,
) -> None:
    result = day_range(
        year=year,
        month=month,
        day=day,
    )

    assert result == expected