from langchain_core.embeddings import Embeddings
from pydantic.dataclasses import dataclass

from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass(frozen=True)
class DocRetrievalConfig:
    top_k: int = 5

class DocHitsRetriever:
    def __init__(
        self,
        *,
        embedder: Embeddings,
        doc_store: DocStore,
        config: DocRetrievalConfig,
    ):
        self._embedder = embedder
        self._doc_store = doc_store
        self._config = config

    def retrieve(
        self,
        *,
        question: str,
    ) -> list[DocHit]:
        query_vector = self._embedder.embed_query(question)
        
        chunk_scores: list[tuple[EmbeddedDocChunk, float]] = [
            (chunk, vectors_cosine_similarity(query_vector, chunk.embedding))
            for chunk in self._doc_store.get_embedded_chunks()
        ]

        sorted_chunks = sorted(
            chunk_scores,
            key=lambda item: item[1],
            reverse=True
        )
        
        doc_hits = [
            DocHit(
                id=f"{chunk.doc_reference.file_name}_{chunk.chunk_id}",
                doc_reference=chunk.doc_reference,
                retrieval_score=score,
                content=chunk.content,
            )
            for chunk, score in sorted_chunks[:self._config.top_k]
        ]
        
        return doc_hits