from decimal import Decimal

from bot.tx_qa.domain import Transaction


def compute_total_amount(transactions: list[Transaction]) -> Decimal:
    total_amount = Decimal("0.00")
    for tx in transactions:
        total_amount += Decimal(str(tx.amount))
    return total_amount
