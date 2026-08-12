from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum


class Label(Enum):
    FOOD = "food"
    PETS = "pets"
    OTHER = "other"
    
class Direction(Enum):
    SPEND = "spend"
    RECEIVE = "receive"

@dataclass(frozen=True)
class Transaction:
    id: str
    date: date
    amount: Decimal
    direction: Direction
    other_account: str
    other_contact_name: str | None = None
    description: str | None = None
    label: Label | None = None
