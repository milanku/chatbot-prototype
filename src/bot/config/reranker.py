from enum import StrEnum

from pydantic import BaseModel


class SupportedLocalReranker(StrEnum):
    BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"
    
class LocalRerankerConfig(BaseModel):
    model: SupportedLocalReranker
    use_fp16: bool
    top_k: int

RerankerConfig = LocalRerankerConfig