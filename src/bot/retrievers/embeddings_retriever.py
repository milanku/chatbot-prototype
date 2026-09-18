from langchain_core.embeddings import Embeddings

from bot.config.retriever import EmbeddingsRetrieverConfig
from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from bot.retrievers.models import ChunksRetriever
from bot.routes.doc_qa.score import vectors_cosine_similarity


class EmbeddingsChunksRetriever(ChunksRetriever):
    def __init__(
        self,
        *,
        embedder: Embeddings,
        embedded_doc_chunks: list[EmbeddedDocChunk],
        config: EmbeddingsRetrieverConfig,
    ):
        self._embedder = embedder
        self._embedded_doc_chunks = embedded_doc_chunks
        self._config = config

    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        query_vector = self._embedder.embed_query(question)
        
        chunk_scores: list[tuple[DocChunk, float]] = [
            (chunk_embedding, vectors_cosine_similarity(query_vector, chunk_embedding.embedding))
            for chunk_embedding in self._embedded_doc_chunks
        ]

        sorted_chunks = sorted(
            chunk_scores,
            key=lambda item: item[1],
            reverse=True
        )
        
        return [
            DocChunk(
                chunk_id=chunk.chunk_id,
                doc_reference=chunk.doc_reference,
                retrieval_score=score,
                content=chunk.content,
            )
            for chunk, score in sorted_chunks[:self._config.top_k]
        ][:self._config.top_k]