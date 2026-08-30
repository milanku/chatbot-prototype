from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference


@dataclass
class DocHit:
    id: str
    content: str
    doc_reference: DocReference
    retrieval_score: float
    reranker_score: float | None = None