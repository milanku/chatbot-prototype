from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DocReference:
    file_name: str
    heading_path: list[str]
    
@dataclass(frozen=True)
class DocChunk:
    doc_reference: DocReference
    content: str
    chunk_id: int
    
@dataclass(frozen=True) 
class DocHit:
    id: str
    score: float
    doc_reference: DocReference
    content: str

@dataclass
class DocRepository(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[str]: ...
    
    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]: ...