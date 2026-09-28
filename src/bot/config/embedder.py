from enum import StrEnum

from pydantic import BaseModel


class SupportedOpenAIEmbedder(StrEnum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class OpenAIEmbedderConfig(BaseModel):
    model: SupportedOpenAIEmbedder


class SupportedLocalEmbedder(StrEnum):
    JINA_EMBEDDINGS_V3 = "jinaai/jina-embeddings-v3"
    QWEN3_EMBEDDING_4B = "qwen/Qwen3-Embedding-4B"


class LocalEmbedderConfig(BaseModel):
    model: SupportedLocalEmbedder


EmbedderConfig = OpenAIEmbedderConfig | LocalEmbedderConfig
