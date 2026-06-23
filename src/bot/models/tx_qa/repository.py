from dataclasses import dataclass
from typing import Protocol

from bot.models.tx_qa.domain import Transaction
from bot.models.tx_qa.query import TxFilter


@dataclass
class TransactionsRepository(Protocol):
    def list_transactions(self, tx_filter: TxFilter) -> list[Transaction]: ...

    def list_transaction_ids(self, tx_filter: TxFilter) -> list[str]: ...