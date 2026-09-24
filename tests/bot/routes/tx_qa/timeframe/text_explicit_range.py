from datetime import date

import pytest

from bot.tx_qa.timeframe.calendar_utils import DateRange
from bot.tx_qa.timeframe.explicit_range import resolve_explicit_range
from bot.tx_qa.timeframe.models import RawRangeEndpoint


# TODO: Update the test when main function is refactored
@pytest.mark.parametrize(
    (
        "start_endpoint",
        "end_endpoint",
        "today",  
        "expected",
    ),
    [
        pytest.param(
            RawRangeEndpoint(year=2021, month=7, day=15),
            RawRangeEndpoint(year=2021, month=7, day=28),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 7, 15),
                end_date=date(2021, 7, 28),
            ),
            id="explicit-range-all-parts",
        ),        
        pytest.param(
            RawRangeEndpoint(month=7, day=15),
            RawRangeEndpoint(year=2021, month=7, day=28),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 7, 15),
                end_date=date(2021, 7, 28),
            ),
            id="explicit-range-infer-year-from-start",
        ),
        pytest.param(
            RawRangeEndpoint(day=15),
            RawRangeEndpoint(year=2021, month=7, day=28),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 7, 15),
                end_date=date(2021, 7, 28),
            ),
            id="explicit-range-infer-year-and-month-from-start",
        ),
        pytest.param(
            RawRangeEndpoint(year=2021, month=7, day=15),
            RawRangeEndpoint(month=7, day=28),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 7, 15),
                end_date=date(2021, 7, 28),
            ),
            id="explicit-range-infer-year-from-end",
        ),
        pytest.param(
            RawRangeEndpoint(year=2021, month=7, day=15),
            RawRangeEndpoint(day=28),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 7, 15),
                end_date=date(2021, 7, 28),
            ),
            id="explicit-range-infer-year-and-month-from-end",
        ),        
        pytest.param(
            RawRangeEndpoint(year=2021, quarter=1),
            RawRangeEndpoint(year=2021, quarter=3),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 9, 30),
            ),
            id="explicit-range-quarter-base",
        ),
        pytest.param(
            RawRangeEndpoint(quarter=1),
            RawRangeEndpoint(year=2021, quarter=3),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 9, 30),
            ),
            id="explicit-range-quarter-infer-start-year",
        ),
        pytest.param(
            RawRangeEndpoint(quarter=1),
            RawRangeEndpoint(quarter=3),
            date(2021, 7, 1),
            DateRange(
                start_date=date(2021, 1, 1),
                end_date=date(2021, 9, 30),
            ),
            id="explicit-range-quarter-infer-current-year",
        ),
    ],
)
def test_day_range(
    start_endpoint: RawRangeEndpoint,
    end_endpoint: RawRangeEndpoint,
    today: date,
    expected: DateRange | None,
) -> None:
    result = resolve_explicit_range(
        start_endpoint=start_endpoint,
        end_endpoint=end_endpoint,
        today=today,
    )

    assert result == expected