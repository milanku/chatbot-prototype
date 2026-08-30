import numpy as np
from langchain.embeddings import Embeddings
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class JinaEmbeddings(Embeddings):
    def __init__(self):
        self._model = SentenceTransformer(
            "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
        )
        
    def embed_query(self, text: str) -> list[float]:
        query_embedding: list[float] = self._model.encode(text, normalize_embeddings=True).tolist() # pyright: ignore[reportUnknownMemberType]
        return query_embedding
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        document_embeddings: NDArray[np.float32] = self._model.encode(texts, normalize_embeddings=True) # pyright: ignore[reportUnknownMemberType]
        return document_embeddings.tolist()