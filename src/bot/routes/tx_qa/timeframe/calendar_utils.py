import calendar
from datetime import date

from bot.models.tx_qa.query import DateRange


def year_range(year: int) -> DateRange:
    return date(year, 1, 1), date(year, 12, 31)

def quarter_range(year: int, quarter: int) -> DateRange:
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid quarter: {quarter}")

    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3

    start_date = date(year, start_month, 1)
    end_date = date(year, end_month, calendar.monthrange(year, end_month)[1])

    return start_date, end_date

def month_range(year: int, month: int) -> DateRange:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)

def day_range(year: int, month: int, day: int) -> DateRange:
    return date(year, month, day), date(year, month, day)