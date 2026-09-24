
from dataclasses import dataclass

from bot.doc_qa.indexing.models import DocReference


@dataclass(frozen=True)
class BotResponse:
    trace_id: str
    answer: str
    doc_references: list[DocReference] # List of DocReference objects