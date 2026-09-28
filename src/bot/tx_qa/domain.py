from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel


class Label(StrEnum):
    FOOD = "food"
    PETS = "pets"
    OTHER = "other"


class Direction(StrEnum):
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
