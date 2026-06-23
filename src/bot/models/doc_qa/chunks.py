from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference


@dataclass(frozen=True)
class DocChunk:
    doc_reference: DocReference
    content: str
    chunk_id: int
    
@dataclass(frozen=True)
class EmbeddedDocChunk(DocChunk):
    embedding: list[float]