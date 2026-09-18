from enum import StrEnum

from pydantic import BaseModel


class OpenAIEmbedderConfig(BaseModel):
    model: str

class SupportedLocalEmbedder(StrEnum):
    JINA_EMBEDDINGS_V3 = "jinaai/jina-embeddings-v3"
    QWEN3_EMBEDDING_4B = "qwen/Qwen3-Embedding-4B"
    
class LocalEmbedderConfig(BaseModel):
    model: SupportedLocalEmbedder

EmbedderConfig = OpenAIEmbedderConfig | LocalEmbedderConfig