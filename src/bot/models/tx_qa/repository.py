from dataclasses import dataclass
from datetime import date
from typing import Protocol

from bot.models.tx_qa.domain import Direction, Label, Transaction


@dataclass(frozen=True)
class TxFilter:
    label: Label | None
    start_date: date
    end_date: date
    direction: Direction | None

class TransactionsRepository(Protocol):
    def list_transactions(self, tx_filter: TxFilter) -> list[Transaction]: ...

    def list_transaction_ids(self, tx_filter: TxFilter) -> list[str]: ...