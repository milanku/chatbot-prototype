from typing import Protocol

from bot.tx_qa.domain import Transaction
from bot.tx_qa.parsing.models import TxQuery


class TransactionsRepository(Protocol):
    def list_transactions(self, tx_query: TxQuery) -> list[Transaction]: ...

    def list_transaction_ids(self, tx_query: TxQuery) -> list[str]: ...
