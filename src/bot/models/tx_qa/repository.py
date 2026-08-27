from typing import Protocol

from bot.models.tx_qa.domain import Transaction
from bot.models.tx_qa.query import TxQuery


class TransactionsRepository(Protocol):
    def list_transactions(self, tx_query: TxQuery) -> list[Transaction]: ...

    def list_transaction_ids(self, tx_query: TxQuery) -> list[str]: ...