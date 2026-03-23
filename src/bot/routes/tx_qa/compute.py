from decimal import Decimal

from bot.models.domain import Transaction


def compute_total_spent(transactions: list[Transaction]) -> Decimal:
    total = Decimal("0.00")
    for tx in transactions:
        if tx.direction == "spend":
            total += Decimal(str(tx.amount))
    return total
