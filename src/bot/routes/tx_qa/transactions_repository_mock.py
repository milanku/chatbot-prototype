import json
from pathlib import Path

from pydantic import ValidationError

from bot.models.tx_qa.domain import Transaction
from bot.models.tx_qa.query import TxQuery
from bot.models.tx_qa.repository import TransactionsRepository


class TransactionsRepositoryFromJsonMock(TransactionsRepository):
    _transactions: list[Transaction]
    
    def __init__(self, *, _transactions: list[Transaction]):
        self._transactions = _transactions

    @classmethod
    def from_json_file(cls, file_path: Path) -> "TransactionsRepositoryFromJsonMock":
        try:
            raw_data = json.loads(file_path.read_text(encoding="utf-8"))
            transactions_data = [
                Transaction.model_validate(tx) for tx in raw_data
            ]
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file at {file_path}:\n {e}")
        except ValidationError as e:
            raise ValueError(f"Invalid transaction data in JSON file at {file_path}:\n {e}")
        
        transactions: list[Transaction] = [
            Transaction(
                id=tx.id,
                date=tx.date,
                amount=tx.amount,
                direction=tx.direction,
                other_account=tx.other_account,
                other_contact_name=tx.other_contact_name,
                description=tx.description,
                label=tx.label
            )
            for tx in transactions_data
        ]
        return cls(_transactions=transactions)

    def list_transactions(self, tx_query: TxQuery) -> list[Transaction]:
        return [
            tx
            for tx in self._transactions
            if tx_query.start_date <= tx.date <= tx_query.end_date
            and tx.direction == tx_query.direction
            and tx.label == tx_query.label
        ]

    def list_transaction_ids(self, tx_query: TxQuery) -> list[str]:
        return [tx.id for tx in self.list_transactions(tx_query)]
