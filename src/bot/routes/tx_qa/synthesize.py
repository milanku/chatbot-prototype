from bot.models.tx_qa.query import TxQAQuery


def synthesize_tx_summary(query: TxQAQuery, total_amount: float) -> str:
    if query.direction == "spend":
        return f"You spent a total of ${total_amount:.2f} on {query.label} between {query.start} and {query.end}."
    elif query.direction == "receive":
        return f"You received a total of ${total_amount:.2f} for {query.label} between {query.start} and {query.end}."
    else:
        return f"You had a total of ${total_amount:.2f} for {query.label} between {query.start} and {query.end}."