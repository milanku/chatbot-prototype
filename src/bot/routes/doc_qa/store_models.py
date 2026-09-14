from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk


class EmbeddingsStoreManifest(BaseModel):
    embedding_model: str
    chunking_version: str
    docs_fingerprint: str
    
class ChunkEmbedding(BaseModel):
    chunk_id: str
    embedding: list[float]
    
class EmbeddingsStoreData(BaseModel):
    manifest: EmbeddingsStoreManifest | None = None
    chunks: dict[str, DocChunk]
    chunk_embeddings: dict[str, ChunkEmbedding]