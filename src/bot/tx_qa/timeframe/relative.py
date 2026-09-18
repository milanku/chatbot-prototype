from datetime import date, timedelta

from bot.tx_qa.timeframe.calendar_utils import month_range, year_range
from bot.tx_qa.timeframe.models import DateRange


def resolve_relative_year(
    offset: int,
    today: date,
) -> DateRange:
    return year_range(today.year + offset)

def resolve_relative_month(
    offset: int,
    today: date,
) -> DateRange:
    # target_month_index: 0 (January) to 11 (December)
    target_month_index = today.month - 1 + offset
    year = today.year + target_month_index // 12
    month = target_month_index % 12 + 1
    return month_range(year, month)

def resolve_relative_week(
    offset: int,
    today: date,
) -> DateRange:
    current_monday = today - timedelta(days=today.weekday())
    start = current_monday + timedelta(weeks=offset)
    return DateRange(start_date=start, end_date=start + timedelta(days=6))

def resolve_relative_day(
    offset: int,
    today: date,
) -> DateRange:
    target_day = today + timedelta(days=offset)
    return DateRange(start_date=target_day, end_date=target_day)
