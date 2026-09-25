from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel


class Label(str, Enum):
    FOOD = "food"
    PETS = "pets"
    OTHER = "other"


class Direction(str, Enum):
    SPEND = "spend"
    RECEIVE = "receive"


class Transaction(BaseModel):
    id: str
    date: date
    amount: Decimal
    direction: Direction
    other_account: str
    other_contact_name: str | None = None
    description: str | None = None
    label: Label | None = None
