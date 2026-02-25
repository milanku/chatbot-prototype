from datetime import date
from decimal import Decimal
from typing import Literal

from attr import dataclass

Direction = Literal["spend", "receive"]
Label = Literal["food", "pets", "other"]


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
