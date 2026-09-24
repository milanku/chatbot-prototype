from pydantic import BaseModel, Field


class BM25RetrieverConfig(BaseModel):
    top_k: int = Field(gt=0)


class EmbeddingsRetrieverConfig(BaseModel):
    top_k: int = Field(gt=0)


RetrieverConfig = BM25RetrieverConfig | EmbeddingsRetrieverConfig
    
