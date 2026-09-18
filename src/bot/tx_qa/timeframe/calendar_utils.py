import calendar
from datetime import date

from bot.tx_qa.timeframe.models import DateRange


def year_range(year: int) -> DateRange:
    return DateRange(start_date=date(year, 1, 1), end_date=date(year, 12, 31))

def quarter_range(year: int, quarter: int) -> DateRange:
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid quarter: {quarter}")

    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3

    start_date = date(year, start_month, 1)
    end_date = date(year, end_month, calendar.monthrange(year, end_month)[1])

    return DateRange(start_date=start_date, end_date=end_date)

def month_range(year: int, month: int) -> DateRange:
    last_day = calendar.monthrange(year, month)[1]
    return DateRange(start_date=date(year, month, 1), end_date=date(year, month, last_day))

def day_range(year: int, month: int, day: int) -> DateRange:
    return DateRange(start_date=date(year, month, day), end_date=date(year, month, day))