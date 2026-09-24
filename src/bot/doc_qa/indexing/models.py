from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class DocReference:
    file_name: str
    heading_path: list[str]

class DocChunk(BaseModel):
    doc_reference: DocReference
    content: str
    chunk_id: str
    retrieval_score: float | None = None
    reranker_score: float | None = None
    
class EmbeddedDocChunk(DocChunk):
    embedding: list[float]
    

class Chunker(Protocol):    
    def split(self, text: str, file_name: str) -> list[DocChunk]:
        ...