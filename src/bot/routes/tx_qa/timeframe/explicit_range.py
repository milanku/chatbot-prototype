from datetime import date, timedelta

from bot.logging import log_event
from bot.models.tx_qa.query import DateRange, RawRangeEndpoint


def resolve_explicit_range(
    start_endpoint: RawRangeEndpoint,
    end_endpoint: RawRangeEndpoint,
    today: date,
) -> DateRange | None:
    start_date: date | None = None
    end_date: date | None = None
    
    start_day = start_endpoint.day
    start_month = start_endpoint.month
    start_year = start_endpoint.year
    start_quarter = start_endpoint.quarter

    end_day = end_endpoint.day
    end_month = end_endpoint.month
    end_year = end_endpoint.year
    end_quarter = end_endpoint.quarter
    
    # TODO: fix very awkward implementation, refactor needed. This is very basic and does not handle all cases.
    # @today might be needed to resolve cases where some parts are meant to be implied
    
    if(start_day is not None):
        if(start_month is not None):
            if(start_year is not None):
                start_date = date(start_year, start_month, start_day)
            elif(end_year is not None):
                start_date = date(end_year, start_month, start_day)
            else:
                candidate_date = date(today.year, start_month, start_day)
                if(candidate_date > today):
                    candidate_date = date(today.year - 1, start_month, start_day)
                start_date = candidate_date
        elif(end_month is not None):
            if(start_year is not None):
                start_date = date(start_year, end_month, start_day)
            elif(end_year is not None):
                start_date = date(end_year, end_month, start_day)
    elif(start_month is not None):
        if(start_year is not None):
            start_date = date(start_year, start_month, 1)
        elif(end_year is not None):
            start_date = date(end_year, start_month, 1)
    elif(start_year is not None):
        start_date = date(start_year, 1, 1)
        
    if(end_day is not None):
        if(end_month is not None):
            if(end_year is not None):
                end_date = date(end_year, end_month, end_day)
            elif(start_year is not None):
                end_date = date(start_year, end_month, end_day)
            else:
                candidate_date = date(today.year, end_month, end_day)
                if(candidate_date > today):
                    candidate_date = date(today.year - 1, end_month, end_day)
                end_date = candidate_date
        elif(start_month is not None):
            if(end_year is not None):
                end_date = date(end_year, start_month, end_day)
            elif(start_year is not None):
                end_date = date(start_year, start_month, end_day)
    elif(end_month is not None):
        if(end_year is not None):
            last_day = (date(end_year, end_month + 1, 1) - timedelta(days=1)).day
            end_date = date(end_year, end_month, last_day)
        elif(start_year is not None):
            last_day = (date(start_year, end_month + 1, 1) - timedelta(days=1)).day
            end_date = date(start_year, end_month, last_day)
    elif(end_year is not None):
        end_date = date(end_year, 12, 31)
        
    if(start_quarter is not None):
        if(start_year is not None):
            start_month = (start_quarter - 1) * 3 + 1
            start_date = date(start_year, start_month, 1)
        elif(end_year is not None):
            start_month = (start_quarter - 1) * 3 + 1
            start_date = date(end_year, start_month, 1)
        else:
            current_year = today.year
            start_month = (start_quarter - 1) * 3 + 1
            start_date = date(current_year, start_month, 1)
            
    if(end_quarter is not None):
        if(end_year is not None):
            end_month = end_quarter * 3
            last_day = (date(end_year, end_month + 1, 1) - timedelta(days=1)).day
            end_date = date(end_year, end_month, last_day)
        elif(start_year is not None):
            end_month = end_quarter * 3
            last_day = (date(start_year, end_month + 1, 1) - timedelta(days=1)).day
            end_date = date(start_year, end_month, last_day)
        else:
            current_year = today.year
            end_month = end_quarter * 3
            last_day = (date(current_year, end_month + 1, 1) - timedelta(days=1)).day
            end_date = date(current_year, end_month, last_day)
            
    log_event(
        event="resolve_explicit_range",
        payload={
            "start_endpoint": start_endpoint.model_dump(),
            "end_endpoint": end_endpoint.model_dump(),
            "today": today.isoformat(),
            "resolved_start_date": start_date.isoformat() if start_date else None,
            "resolved_end_date": end_date.isoformat() if end_date else None,
        }
    )


    return (start_date, end_date) if start_date is not None and end_date is not None else None