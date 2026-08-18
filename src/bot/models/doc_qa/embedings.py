from pydantic import BaseModel


class EmbeddingsManifest(BaseModel):
    embedding_model: str
    chunking_version: str
    docs_fingerprint: str
    chunk_count: int
    
