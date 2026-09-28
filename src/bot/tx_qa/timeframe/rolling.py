import calendar
from datetime import date, timedelta

from bot.tx_qa.timeframe.models import DateRange, RollingRangeUnit


# Calculates the trailing range (past X time units ending with today) based on the specified unit and amount. It returns a DateRange object representing the start and end dates of the trailing range. Includes current day (last 5 days = today and 4 days before)
def resolve_trailing_range(
    *, unit: RollingRangeUnit, unit_amount: int, today: date
) -> DateRange | None:
    match unit:
        case RollingRangeUnit.DAY:
            start_date = today - timedelta(days=unit_amount - 1)
            return DateRange(start_date=start_date, end_date=today)
        case RollingRangeUnit.WEEK:
            start_date = today - timedelta(days=7 * unit_amount - 1)
            return DateRange(start_date=start_date, end_date=today)
        case RollingRangeUnit.MONTH:
            start_month_index = today.month - 1 - unit_amount
            start_month = (start_month_index) % 12 + 1
            start_year = today.year + (start_month_index // 12)
            last_day_of_start_month = calendar.monthrange(start_year, start_month)[1]
            return DateRange(
                start_date=date(start_year, start_month, min(today.day, last_day_of_start_month)),
                end_date=today,
            )
        case RollingRangeUnit.QUARTER:
            return None  # Trailing range for quarters is not implemented
        case RollingRangeUnit.YEAR:
            start_year = today.year - unit_amount
            last_day_of_start_month = calendar.monthrange(start_year, today.month)[1]
            return DateRange(
                start_date=date(start_year, today.month, min(today.day, last_day_of_start_month)),
                end_date=today,
            )
    return None  # If the unit is not recognized, return None


# Calculates the previous complete range (past X time units excluding current unit (e.g. previous 5 months means 5 months without current month)) based on the specified unit and amount. It returns a DateRange object representing the start and end dates of the previous complete range.
def resolve_previous_complete_range(
    *, unit: RollingRangeUnit, unit_amount: int, today: date
) -> DateRange | None:
    match unit:
        case RollingRangeUnit.DAY:
            start_date = today - timedelta(days=unit_amount)
            end_date = today - timedelta(days=1)
            return DateRange(start_date=start_date, end_date=end_date)
        case RollingRangeUnit.WEEK:
            current_monday = today - timedelta(days=today.weekday())
            start_date = current_monday - timedelta(weeks=unit_amount)
            end_date = current_monday - timedelta(days=1)
            return DateRange(start_date=start_date, end_date=end_date)
        case RollingRangeUnit.MONTH:
            start_month_index = today.month - 1 - unit_amount
            start_month = (start_month_index) % 12 + 1
            start_year = today.year + (start_month_index // 12)
            end_month_index = today.month - 2
            end_month = (end_month_index) % 12 + 1
            end_year = today.year + (end_month_index // 12)
            last_day_of_end_month = calendar.monthrange(end_year, end_month)[1]
            return DateRange(
                start_date=date(start_year, start_month, 1),
                end_date=date(end_year, end_month, last_day_of_end_month),
            )
        case RollingRangeUnit.QUARTER:
            current_quarter = (today.month - 1) // 3 + 1
            start_quarter = (current_quarter - unit_amount - 1) % 4 + 1
            start_year = today.year + (current_quarter - unit_amount - 1) // 4
            end_quarter = (current_quarter - 2) % 4 + 1
            end_year = today.year + (current_quarter - 2) // 4
            start_month = (start_quarter - 1) * 3 + 1
            end_month = end_quarter * 3
            last_day_of_end_month = calendar.monthrange(end_year, end_month)[1]
            return DateRange(
                start_date=date(start_year, start_month, 1),
                end_date=date(end_year, end_month, last_day_of_end_month),
            )
        case RollingRangeUnit.YEAR:
            start_year = today.year - unit_amount
            end_year = today.year - 1
            return DateRange(start_date=date(start_year, 1, 1), end_date=date(end_year, 12, 31))
    return None  # If the unit is not recognized, return None
