from datetime import date
from decimal import Decimal

import pytest

from bot.tx_qa.domain import Direction, Transaction
from bot.tx_qa.compute import compute_total_amount


def get_transaction(amount: str) -> Transaction:
    return Transaction(
        amount=Decimal(amount),
        date=date(2024, 1, 1),
        description="Test transaction",
        id="test_id",
        direction=Direction.SPEND,
        other_account="Test Account",    
    )

@pytest.mark.parametrize(
    "transactions, expected_total",
    [
        ([], Decimal("0.00")),
        ([get_transaction(amount="10")], Decimal("10.00")),
        ([get_transaction(amount="10"), get_transaction(amount="20.5")], Decimal("30.50")),
        ([get_transaction(amount="10.33"), get_transaction(amount="20"), get_transaction(amount="-50")], Decimal("-19.67")),
    ]
)

def test_compute_total_amount(transactions: list[Transaction], expected_total: Decimal) -> None:

    total = compute_total_amount(transactions)
    assert total == expected_total