from typing import Protocol

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic.dataclasses import dataclass

from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5

class ChunksRetriever(Protocol):
    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        ...

class EmbeddingsChunksRetriever(ChunksRetriever):
    def __init__(
        self,
        *,
        embedder: Embeddings,
        embedded_doc_chunks: list[EmbeddedDocChunk],
        config: RetrievalConfig,
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
        ]
        
    
class BM25ChunksRetriever(ChunksRetriever):
    def __init__(
        self,
        *,
        embedded_doc_chunks: list[EmbeddedDocChunk],
        config: RetrievalConfig,
    ):
        self._doc_chunks = embedded_doc_chunks
        self._config = config

    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        bm25retriever = BM25Retriever.from_documents(
            documents=[
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id
                    }
                ) for chunk in self._doc_chunks
            ]
        )
        doc_chunks_by_content = {chunk.content: chunk for chunk in self._doc_chunks}
        bm25retriever.k = self._config.top_k
        candidate_chunks_from_bm25 = bm25retriever.invoke(question)
        return [
            DocChunk(
                chunk_id=chunk.metadata['chunk_id'],
                content=chunk.page_content,
                doc_reference=doc_chunks_by_content[chunk.page_content].doc_reference)
            for chunk in candidate_chunks_from_bm25
        ]