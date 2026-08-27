from datetime import date

import pytest

from bot.models.tx_qa.query import DateRange
from bot.routes.tx_qa.timeframe.weekday import resolve_named_weekday


@pytest.mark.parametrize(
    (
        "weekday",
        "offset",
        "today",
        "expected",
    ),
    [
        pytest.param(
            1,
            -1,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 27),
                end_date=date(2026, 7, 27),
            ),
            id="named-weekday-offset-same-week",
        ),
        pytest.param(
            6,
            -1,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 25),
                end_date=date(2026, 7, 25),
            ),
            id="named-weekday-offset-previous-week",
        ),
        pytest.param(
            1,
            0,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 27),
                end_date=date(2026, 7, 27),
            ),
            id="named-weekday-offset-same-week-2",
        ),
        pytest.param(
            6,
            0,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 25),
                end_date=date(2026, 7, 25),
            ),
            id="named-weekday-offset-previous-week-2",
        ),       
        pytest.param(
            1,
            -2,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 20),
                end_date=date(2026, 7, 20),
            ),
            id="named-weekday-double-offset-previous-week",
        ),
        pytest.param(
            6,
            -2,
            date(2026, 7, 30),
            DateRange(
                start_date=date(2026, 7, 18),
                end_date=date(2026, 7, 18),
            ),
            id="named-weekday-double-offset-previous-week-2",
        ),
    ],
)
def test_range_from_named_weekday(
    weekday: int,
    offset: int | None,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_named_weekday(
        weekday=weekday,
        offset=offset,
        today=today
    )

    assert result == expected