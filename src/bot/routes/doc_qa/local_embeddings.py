from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray
from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(BaseRagasEmbeddings, Embeddings):
    def __init__(self, sentence_transformer: SentenceTransformer):
        self._model = sentence_transformer
        
    def embed_query(self, text: str) -> list[float]:
        query_embedding: list[float] = self._model.encode(
            text,
            task="retrieval.query",
            normalize_embeddings=True,
        ).tolist()  # pyright: ignore[reportUnknownMemberType]
        return query_embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        document_embeddings: NDArray[np.float32] = self._model.encode(
            texts,
            task="retrieval.passage",
            normalize_embeddings=True,
        )  # pyright: ignore[reportUnknownMemberType]
        return document_embeddings.tolist()
    
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)