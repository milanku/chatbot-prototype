from decimal import Decimal

from bot.tx_qa.domain import Direction
from bot.tx_qa.parsing.models import TxQuery


def format_tx_summary(query: TxQuery, total_amount: Decimal) -> str:
    if query.direction == Direction.SPEND:
        return f"You spent a total of ${total_amount:.2f} on {query.label} between {query.start_date} and {query.end_date}."
    elif query.direction == Direction.RECEIVE:
        return f"You received a total of ${total_amount:.2f} for {query.label} between {query.start_date} and {query.end_date}."
    else:
        return f"You had a total of ${total_amount:.2f} for {query.label} between {query.start_date} and {query.end_date}."
