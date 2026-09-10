from pydantic import BaseModel

from bot.models.doc_qa.references import DocReference


class DocChunk(BaseModel):
    doc_reference: DocReference
    content: str
    chunk_id: str
    retrieval_score: float | None = None
    reranker_score: float | None = None
    
class EmbeddedDocChunk(DocChunk, BaseModel):
    embedding: list[float]