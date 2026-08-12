import calendar
from datetime import date, timedelta

from bot.models.tx_qa.query import DateRange, RollingRangeUnit


# Calculates the trailing range based on the specified unit and amount. It returns a DateRange object representing the start and end dates of the trailing range. Includes current day (last 5 days = today and 4 days before)
def resolve_trailing_range(
    *,
    unit: RollingRangeUnit,
    amount: int,
    today: date
) -> DateRange | None:
    match unit:
        case RollingRangeUnit.DAY:
            start = today - timedelta(days=amount - 1)
            return start, today
        case RollingRangeUnit.WEEK:
            start = today - timedelta(days=7 * amount - 1)
            return start, today
        case RollingRangeUnit.MONTH:
            start_month = (today.month - amount - 1) % 12 + 1
            start_year = today.year + (today.month - amount - 1) // 12
            start_month_last_day = calendar.monthrange(start_year, start_month)[1]
            return date(start_year, start_month, min(today.day, start_month_last_day)), today
        case RollingRangeUnit.QUARTER:
            return None  # Trailing range for quarters is not implemented
        case RollingRangeUnit.YEAR:
            start_year = today.year - amount
            start_month_last_day = calendar.monthrange(start_year, today.month)[1]
            return date(start_year, today.month, min(today.day, start_month_last_day)), today
    return None  # If the unit is not recognized, return None

def resolve_previous_complete_range(
    *,
    unit: RollingRangeUnit,
    amount: int,
    today: date
) -> DateRange | None:
    match unit:
        case RollingRangeUnit.DAY:
            start_day = today - timedelta(days=amount)
            end_day = today - timedelta(days=1)
            return start_day, end_day
        case RollingRangeUnit.WEEK:
            current_monday = today - timedelta(days=today.weekday())
            start_day = current_monday - timedelta(weeks=amount)
            end_day = current_monday - timedelta(days=1)
            return start_day, end_day
        case RollingRangeUnit.MONTH:
            start_month = (today.month - amount - 1) % 12 + 1
            start_year = today.year + (today.month - amount - 1) // 12
            end_month = (today.month - 2) % 12 + 1
            end_year = today.year + (today.month - 2) // 12
            last_day_of_end_month = calendar.monthrange(end_year, end_month)[1]
            return date(start_year, start_month, 1), date(end_year, end_month, last_day_of_end_month)
        case RollingRangeUnit.QUARTER:
            current_quarter = (today.month - 1) // 3 + 1
            start_quarter = (current_quarter - amount - 1) % 4 + 1
            start_year = today.year + (current_quarter - amount - 1) // 4
            end_quarter = (current_quarter - 2) % 4 + 1
            end_year = today.year + (current_quarter - 2) // 4
            start_month = (start_quarter - 1) * 3 + 1
            end_month = end_quarter * 3
            last_day_of_end_month = calendar.monthrange(end_year, end_month)[1]
            return date(start_year, start_month, 1), date(end_year, end_month, last_day_of_end_month)
        case RollingRangeUnit.YEAR:
            start_year = today.year - amount
            end_year = today.year - 1
            return date(start_year, 1, 1), date(end_year, 12, 31)
    return None  # If the unit is not recognized, return None