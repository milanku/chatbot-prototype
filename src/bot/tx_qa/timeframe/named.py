from datetime import date

from bot.tx_qa.timeframe.calendar_utils import (
    day_range,
    month_range,
    quarter_range,
    year_range,
)
from bot.tx_qa.timeframe.models import DateRange


def resolve_named_year(
    *,
    year: int,
) -> DateRange:
    return year_range(year)


def resolve_named_quarter(
    *,
    year: int | None,
    quarter: int,
    today: date,
) -> DateRange | None:
    # If year isn't provided, choose the most recent occurrence of the quarter in current/previous year
    target_year = year
    if target_year is None:
        current_quarter = (today.month - 1) // 3 + 1
        if quarter <= current_quarter:
            target_year = today.year
        else:
            target_year = today.year - 1

    return quarter_range(target_year, quarter)


def resolve_named_month(
    *,
    year: int | None,
    month: int,
    today: date,
) -> DateRange:
    # If year isn't provided, choose the most recent occurrence of the month in current/previous year
    target_year = year
    if target_year is None:
        if month <= today.month:
            target_year = today.year
        else:
            target_year = today.year - 1
    return month_range(target_year, month)


# Explicit date is specified by a single date, e.g., "January 1, 2026" or "2026-01-01" or "22nd of February" or "on the 3rd"
def resolve_named_date(
    *, year: int | None = None, month: int | None = None, day: int, today: date
) -> DateRange | None:
    target_year = year
    target_month = month
    if year is None:
        if month is None:
            # If only day is provided, choose the most recent occurrence of the day in current/previous month
            target_month = today.month
            target_year = today.year
            if day > today.day:
                target_month -= 1
                if target_month == 0:
                    target_month = 12
                    target_year -= 1
        else:
            # If month is provided but year is not, choose the most recent occurrence of the month in current/previous year
            target_year = today.year
            if month > today.month:
                target_year -= 1

    # Validate final date
    try:
        return (
            day_range(target_year, target_month, day)
            if target_year and target_month and day
            else None
        )
    except Exception:
        return None
