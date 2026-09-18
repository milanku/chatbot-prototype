from pydantic import BaseModel


class BM25RetrieverConfig(BaseModel):
    top_k: int
    
class EmbeddingsRetrieverConfig(BaseModel):
    top_k: int
    
RetrieverConfig = BM25RetrieverConfig | EmbeddingsRetrieverConfig
    
