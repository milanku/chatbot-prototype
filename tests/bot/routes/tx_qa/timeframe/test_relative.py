from datetime import date

import pytest

from bot.models.tx_qa.query import DateRange
from bot.routes.tx_qa.timeframe.relative import (
    resolve_relative_day,
    resolve_relative_month,
    resolve_relative_week,
    resolve_relative_year,
)


@pytest.mark.parametrize(
    (
        "offset",
        "today",
        "expected",
    ),
    [
        pytest.param(
            5,
            date(2030, 6, 20),
            DateRange(
                start_date=date(2035, 1, 1),
                end_date=date(2035, 12, 31),
            ),
            id="relative-year-offset",
        ),
        pytest.param(
            -3,
            date(2025, 1, 15),
            DateRange(
                start_date=date(2022, 1, 1),
                end_date=date(2022, 12, 31),
            ),
            id="relative-year-negative-offset",
        ),
    ],
)
def test_range_from_relative_year(
    offset: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_relative_year(
        offset,
        today
    )

    assert result == expected
    
    
@pytest.mark.parametrize(
    (
        "offset",
        "today",
        "expected",
    ),
    [
        pytest.param(
            5,
            date(2030, 6, 20),
            DateRange(
                start_date=date(2030, 11, 1),
                end_date=date(2030, 11, 30),
            ),
            id="relative-month-offset",
        ),
        pytest.param(
            -3,
            date(2025, 1, 15),
            DateRange(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 10, 31),
            ),
            id="relative-month-negative-offset-previous-year",
        ),
        pytest.param(
            7,
            date(2025, 6, 15),
            DateRange(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 31),
            ),
            id="relative-month-positive-offset-next-year",
        ),
    ],
)
def test_range_from_relative_month(
    offset: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_relative_month(
        offset,
        today
    )

    assert result == expected
    
@pytest.mark.parametrize(
    (
        "offset",
        "today",
        "expected",
    ),
    [
        pytest.param(
            2,
            date(2030, 6, 20),
            DateRange(
                start_date=date(2030, 7, 1),
                end_date=date(2030, 7, 7),
            ),
            id="relative-week-offset",
        ),
        pytest.param(
            -3,
            date(2025, 1, 15),
            DateRange(
                start_date=date(2024, 12, 23),
                end_date=date(2024, 12, 29),
            ),
            id="relative-week-negative-offset-previous-year",
        ),
        pytest.param(
            7,
            date(2025, 12, 15),
            DateRange(
                start_date=date(2026, 2, 2),
                end_date=date(2026, 2, 8),
            ),
            id="relative-week-positive-offset-next-year",
        ),
    ],
)
def test_range_from_relative_week(
    offset: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_relative_week(
        offset,
        today
    )

    assert result == expected
    
@pytest.mark.parametrize(
    (
        "offset",
        "today",
        "expected",
    ),
    [
        pytest.param(
            2,
            date(2025, 7, 20),
            DateRange(
                start_date=date(2025, 7, 22),
                end_date=date(2025, 7, 22),
            ),
            id="relative-day-offset",
        ),
        pytest.param(
            -3,
            date(2025, 1, 3),
            DateRange(
                start_date=date(2024, 12, 31),
                end_date=date(2024, 12, 31),
            ),
            id="relative-day-negative-offset-previous-year",
        ),
        pytest.param(
            7,
            date(2025, 12, 30),
            DateRange(
                start_date=date(2026, 1, 6),
                end_date=date(2026, 1, 6),
            ),
            id="relative-day-positive-offset-next-year",
        ),
    ],
)
def test_range_from_relative_day(
    offset: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_relative_day(
        offset,
        today
    )

    assert result == expected