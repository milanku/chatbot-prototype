from typing import cast

import numpy as np
from numpy.typing import NDArray
from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(BaseRagasEmbeddings):  # type: ignore[misc]
    def __init__(self, sentence_transformer: SentenceTransformer):
        self._transformer = sentence_transformer

    def embed_query(self, text: str) -> list[float]:
        embedding = self._transformer.encode(
            text,
            task="retrieval.query",
            normalize_embeddings=True,
        )

        return cast(list[float], embedding.tolist())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        document_embeddings: NDArray[np.float32] = self._transformer.encode(
            texts,
            task="retrieval.passage",
            normalize_embeddings=True,
        )  # pyright: ignore[reportUnknownMemberType]
        return cast(list[list[float]], document_embeddings.tolist())

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
