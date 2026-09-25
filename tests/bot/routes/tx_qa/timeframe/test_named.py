from datetime import date

import pytest

from bot.tx_qa.timeframe.calendar_utils import DateRange
from bot.tx_qa.timeframe.named import (
    resolve_named_date,
    resolve_named_month,
    resolve_named_quarter,
    resolve_named_year,
)


@pytest.mark.parametrize(
    (
        "year",
        "expected",
    ),
    [
        pytest.param(
            2026,
            DateRange(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 12, 31),
            ),
            id="explicit-year",
        ),
    ],
)
def test_named_year(
    year: int,
    expected: DateRange | None,
) -> None:
    result = resolve_named_year(
        year=year,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("year", "quarter", "today", "expected"),
    [
        pytest.param(
            2024,
            2,
            date(2030, 7, 15),
            DateRange(
                start_date=date(2024, 4, 1),
                end_date=date(2024, 6, 30),
            ),
            id="explicit-year-and-quarter",
        ),
        pytest.param(
            None,
            2,
            date(2026, 7, 15),
            DateRange(
                start_date=date(2026, 4, 1),
                end_date=date(2026, 6, 30),
            ),
            id="past-quarter-uses-current-year",
        ),
        pytest.param(
            None,
            4,
            date(2026, 7, 15),
            DateRange(
                start_date=date(2025, 10, 1),
                end_date=date(2025, 12, 31),
            ),
            id="future-quarter-uses-previous-year",
        ),
    ],
)
def test_named_quarter(
    year: int | None,
    quarter: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_named_quarter(
        year=year,
        quarter=quarter,
        today=today,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("year", "month", "today", "expected"),
    [
        pytest.param(
            2024,
            2,
            date(2030, 7, 15),
            DateRange(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 29),
            ),
            id="explicit-leap-year-month",
        ),
        pytest.param(
            2025,
            2,
            date(2030, 7, 15),
            DateRange(
                start_date=date(2025, 2, 1),
                end_date=date(2025, 2, 28),
            ),
            id="explicit-non-leap-year-month",
        ),
        pytest.param(
            None,
            6,
            date(2026, 7, 15),
            DateRange(
                start_date=date(2026, 6, 1),
                end_date=date(2026, 6, 30),
            ),
            id="past-month-uses-current-year",
        ),
        pytest.param(
            None,
            12,
            date(2026, 7, 15),
            DateRange(
                start_date=date(2025, 12, 1),
                end_date=date(2025, 12, 31),
            ),
            id="future-month-uses-previous-year",
        ),
    ],
)
def test_named_month(
    year: int | None,
    month: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_named_month(
        year=year,
        month=month,
        today=today,
    )

    assert result == expected


@pytest.mark.parametrize(
    (
        "year",
        "month",
        "day",
        "today",
        "expected",
    ),
    [
        pytest.param(
            2026,
            1,
            15,
            date(2030, 6, 20),
            DateRange(
                start_date=date(2026, 1, 15),
                end_date=date(2026, 1, 15),
            ),
            id="explicit-year-month-day",
        ),
        pytest.param(
            None,
            1,
            20,
            date(2026, 2, 5),
            DateRange(
                start_date=date(2026, 1, 20),
                end_date=date(2026, 1, 20),
            ),
            id="month-and-day-use-current-year",
        ),
        pytest.param(
            None,
            3,
            20,
            date(2026, 2, 5),
            DateRange(
                start_date=date(2025, 3, 20),
                end_date=date(2025, 3, 20),
            ),
            id="future-month-and-day-use-previous-year",
        ),
        pytest.param(
            None,
            None,
            2,
            date(2026, 2, 5),
            DateRange(
                start_date=date(2026, 2, 2),
                end_date=date(2026, 2, 2),
            ),
            id="day-already-passed-use-current-month",
        ),
        pytest.param(
            None,
            None,
            20,
            date(2026, 2, 5),
            DateRange(
                start_date=date(2026, 1, 20),
                end_date=date(2026, 1, 20),
            ),
            id="future-day-use-previous-month",
        ),
        pytest.param(
            None,
            None,
            20,
            date(2026, 1, 5),
            DateRange(
                start_date=date(2025, 12, 20),
                end_date=date(2025, 12, 20),
            ),
            id="previous-month-crosses-year-boundary",
        ),
        pytest.param(
            2024,
            2,
            29,
            date(2030, 6, 20),
            DateRange(
                start_date=date(2024, 2, 29),
                end_date=date(2024, 2, 29),
            ),
            id="valid-leap-day",
        ),
        pytest.param(
            2025,
            2,
            29,
            date(2030, 6, 20),
            None,
            id="invalid-leap-day",
        ),
        pytest.param(
            2026,
            None,
            15,
            date(2030, 6, 20),
            None,
            id="explicit-year-but-missing-month",
        ),
        pytest.param(
            2026,
            13,
            1,
            date(2030, 6, 20),
            None,
            id="invalid-month",
        ),
        pytest.param(
            2026,
            4,
            31,
            date(2030, 6, 20),
            None,
            id="invalid-day-for-month",
        ),
    ],
)
def test_range_from_named_date(
    year: int | None,
    month: int | None,
    day: int,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_named_date(
        year=year,
        month=month,
        day=day,
        today=today,
    )

    assert result == expected
