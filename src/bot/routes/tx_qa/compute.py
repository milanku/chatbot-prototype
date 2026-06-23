from decimal import Decimal

from bot.models.tx_qa.domain import Transaction


def compute_total_amount(transactions: list[Transaction]) -> Decimal:
    total = Decimal("0.00")
    for tx in transactions:
        total += Decimal(str(tx.amount))
    return total
