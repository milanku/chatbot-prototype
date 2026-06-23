from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference


@dataclass(frozen=True) 
class DocHit:
    id: str
    score: float
    doc_reference: DocReference
    content: str