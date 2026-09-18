from pydantic import BaseModel


class ContextualChunkerConfig(BaseModel):
    version: str = "contextual_chunker_v01"

ChunkerConfig = ContextualChunkerConfig