from dataclasses import dataclass
from datetime import date
from typing import Protocol

from bot.models.domain import Direction, Label, Transaction


@dataclass(frozen=True)
class TxFilter:
    start: date
    end: date
    direction: Direction
    label: Label


@dataclass
class TransactionsRepository(Protocol):
    def list_transactions(self, tx_filter: TxFilter) -> list[Transaction]: ...

    def list_transaction_ids(self, tx_filter: TxFilter) -> list[str]: ...

@dataclass(frozen=True)
class DocChunk:
    file_name: str
    content: str
    chunk_id: int

@dataclass(frozen=True)
class DocHit:
    id: str
    score: float
    content: str

@dataclass
class DocRepository(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[str]: ...
    
    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]: ...