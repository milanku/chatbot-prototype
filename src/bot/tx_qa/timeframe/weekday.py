from datetime import date, timedelta

from bot.tx_qa.timeframe.models import DateRange


def resolve_named_weekday(
    weekday: int,
    offset: int | None,
    today: date,
) -> DateRange | None:
    if offset == 0 or offset == -1:
        # Most recent occurrence of the weekday in the current or previous week
        current_weekday = today.isoweekday()
        days_difference = (current_weekday - weekday) % 7
        target_date = today - timedelta(days=days_difference)
        return DateRange(start_date=target_date, end_date=target_date)
    elif(offset == -2):
        # Most recent occurrence of the weekday in the week before the previous week
        current_weekday = today.isoweekday()
        days_difference = (current_weekday - weekday) % 7
        target_date = today - timedelta(days=days_difference + 7)
        return DateRange(start_date=target_date, end_date=target_date)
    else:
        return None  # If the offset is not recognized, return None