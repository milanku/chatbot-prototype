from datetime import date
from typing import Protocol

from attr import dataclass

from bot.models.domain import Direction, Label, Transaction


@dataclass(frozen=True)
class TxFilter:
    start: date
    end: date
    direction: Direction
    label: Label


@dataclass
class TransactionsRepository(Protocol):
    def list_transactions(self, filter: TxFilter) -> list[Transaction]: ...

    def list_transaction_ids(self, filter: TxFilter) -> list[str]: ...
