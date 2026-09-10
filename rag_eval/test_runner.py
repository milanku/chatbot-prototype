from FlagEmbedding.inference import FlagReranker
from langchain.embeddings import Embeddings
from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from bot.routes.doc_qa.reranker import CrossEncoderReranker, Reranker
from bot.routes.doc_qa.retriever import (
    ChunksRetriever,
    EmbeddingsChunksRetriever,
    RetrievalConfig,
)
from rag_eval.domain.question import Question
from rag_eval.factories.retrievers import get_retriever


class TestRunnerResult(BaseModel):
    question_id: str
    retrieved_doc_chunks: list[DocChunk]
    bm25_retrieved_doc_chunks: list[DocChunk]
    reranked_doc_chunks: list[DocChunk]

class TestRunner:
    def __init__(
        self,
        retriever: ChunksRetriever,
        bm25_retriever: ChunksRetriever,
        reranker: Reranker,
    ):
        self._retriever = retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker

    def run_test_case(
        self,
        question: Question,
    ) -> TestRunnerResult:
        retrieved_doc_chunks = self._retriever.retrieve(question=question.content)
        bm25_retrieved_doc_chunks = self._bm25_retriever.retrieve(question=question.content)
        
        # Combine the retrieved doc chunks from both retrievers before reranking and remove duplicates
        combined_doc_chunks = retrieved_doc_chunks + bm25_retrieved_doc_chunks
        deduplicated_doc_chunks = list({chunk.chunk_id: chunk for chunk in combined_doc_chunks}.values())
        
        reranked_doc_chunks = self._reranker.rerank(question.content, deduplicated_doc_chunks)
        return TestRunnerResult(
            question_id=question.id,
            retrieved_doc_chunks=retrieved_doc_chunks,
            bm25_retrieved_doc_chunks=bm25_retrieved_doc_chunks,
            reranked_doc_chunks=reranked_doc_chunks
        )
    
    def run_test_suite(
        self,
        questions: list[Question],
    ) -> dict[str, TestRunnerResult]:
        results: dict[str, TestRunnerResult] = {}
        for question in questions:
            results[question.id] = self.run_test_case(question=question)
        return results

def create_test_runner(
    embedder: Embeddings,
    embedded_doc_chunks: list[EmbeddedDocChunk],
    max_top_k: int
) -> TestRunner:
    
    retriever = EmbeddingsChunksRetriever(
        embedder=embedder,
        embedded_doc_chunks=embedded_doc_chunks,
        config=RetrievalConfig(top_k=max_top_k),
    )
    
    reranker = CrossEncoderReranker(
        reranker=FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
        )
    )
    
    bm25_retriever = get_retriever(
        retriever_model="bm25",
        embedded_doc_chunks=embedded_doc_chunks,
        config=RetrievalConfig(top_k=max_top_k)
    )
    
    return TestRunner(
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker
    )