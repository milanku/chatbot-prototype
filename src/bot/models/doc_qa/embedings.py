from dataclasses import dataclass

from bot.models.doc_qa.docs import DocChunk


@dataclass(frozen=True)
class EmbeddedDocChunk(DocChunk):
    embedding: list[float]

@dataclass(frozen=True)
class EmbeddingsManifest:
    embedding_model: str
    docs_fingerprint: str
    chunk_count: int
    
