import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from bot.models.tx_qa.domain import Direction, Label, Transaction
from bot.models.tx_qa.repository import TransactionsRepository, TxFilter


def _parse_date(date_str: str) -> date:
    y, m, d = date_str.split("-")
    return date(int(y), int(m), int(d))


@dataclass
class JsonMockTransactionsRepository(TransactionsRepository):
    _transactions: list[Transaction]

    @classmethod
    def from_json_file(cls, file_path: Path) -> "JsonMockTransactionsRepository":
        raw_data = json.loads(file_path.read_text(encoding="utf-8"))
        transactions: list[Transaction] = []
        for tx in raw_data:
            transactions.append(
                Transaction(
                    id=str(tx["id"]),
                    date=_parse_date(tx["date"]),
                    amount=Decimal(str(tx["amount"])),
                    direction=Direction(tx["direction"]),
                    other_account=tx["other_account"],
                    other_contact_name=tx.get("other_contact_name"),
                    description=tx.get("description"),
                    label=Label(tx["label"]) if tx.get("label") else None,
                )
            )
        return cls(_transactions=transactions)

    def list_transactions(self, tx_filter: TxFilter) -> list[Transaction]:
        return [
            tx
            for tx in self._transactions
            if tx_filter.start <= tx.date <= tx_filter.end
            and tx.direction == tx_filter.direction
            and tx.label == tx_filter.label
        ]

    def list_transaction_ids(self, tx_filter: TxFilter) -> list[str]:
        return [tx.id for tx in self.list_transactions(tx_filter)]
