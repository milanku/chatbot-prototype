import asyncio

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
from rag_eval.config import RunnerConfig
from rag_eval.domain.question import Question
from rag_eval.factories.llms import get_llm
from rag_eval.factories.retrievers import get_retriever
from rag_eval.judges.two_way_chunk_relevance_judge import (
    JudgedChunkOutput,
    TwoWayChunkRelevance,
    TwoWayChunkRelevanceJudge,
    TwoWayRelevanceJudgePromptLoader,
)


class TestRunnerResult(BaseModel):
    question_id: str
    retrieved_chunks: list[DocChunk]
    bm25_retrieved_chunks: list[DocChunk]
    reranked_chunks: list[DocChunk]
    chunks_judged_as_required: list[JudgedChunkOutput]

class TestRunnerResults(BaseModel):
    results: list[TestRunnerResult]

class RetrievalRunner:
    def __init__(
        self,
        retriever: ChunksRetriever,
        bm25_retriever: ChunksRetriever,
        reranker: Reranker,
        two_way_chunk_relevance_judge: TwoWayChunkRelevanceJudge
    ):
        self._retriever = retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._two_way_chunk_relevance_judge = two_way_chunk_relevance_judge

    async def run_test_case(
        self,
        question: Question,
    ) -> TestRunnerResult:
        embedding_retrieved_chunks = self._retriever.retrieve(question=question.content)
        bm25_retrieved_chunks = self._bm25_retriever.retrieve(question=question.content)
        
        retrieved_chunks: list[DocChunk] = []
        retrieved_chunks.extend(embedding_retrieved_chunks)
        retrieved_chunks.extend(bm25_retrieved_chunks)
        
        # Combine the retrieved doc chunks from both retrievers before reranking and remove duplicates
        retrieved_chunks = list({chunk.chunk_id: chunk for chunk in retrieved_chunks}.values())
        reranked_chunks = self._reranker.rerank(question.content, retrieved_chunks)
        
        # Create a simplified version of the reranked chunks with simple chunk IDs for LLM judgment
        chunks_with_simple_ids = [
            DocChunk(
                chunk_id=f"chunk_{i}",
                content=chunk.content,
                doc_reference=chunk.doc_reference)
            for i, chunk in enumerate(reranked_chunks)
        ]
        id_map = {chunk.chunk_id: reranked_chunks[i].chunk_id for i, chunk in enumerate(chunks_with_simple_ids)}
        
        judged_doc_chunks_all = await self._two_way_chunk_relevance_judge.ajudge_chunks(
            question=question.content,
            candidate_chunks=chunks_with_simple_ids
        )
        
        # Map judged chunks back to the original reranked doc chunks
        judged_reranked_chunks: list[JudgedChunkOutput] = [
            JudgedChunkOutput(
                chunk_id=id_map[chunk.chunk_id],
                relevance=chunk.relevance,
                reason=chunk.reason
            )
            for chunk in judged_doc_chunks_all
        ]
        chunks_judged_as_required = [chunk for chunk in judged_reranked_chunks if chunk.relevance == TwoWayChunkRelevance.REQUIRED]
        
        return TestRunnerResult(
            question_id=question.id,
            retrieved_chunks=embedding_retrieved_chunks,
            bm25_retrieved_chunks=bm25_retrieved_chunks,
            reranked_chunks=reranked_chunks,
            chunks_judged_as_required=chunks_judged_as_required
        )
    
    async def _run_test_suite_async(
        self,
        questions: list[Question],
    ) -> TestRunnerResults:
        semaphore = asyncio.Semaphore(10)

        async def run_one(question: Question) -> TestRunnerResult:
            async with semaphore:
                return await self.run_test_case(question)

        results = await asyncio.gather(
            *(run_one(question) for question in questions)
        )

        return TestRunnerResults(
            results=list(results)
        )
    
    def run_test_suite(
        self,
        questions: list[Question],
    ) -> TestRunnerResults:
        return asyncio.run(
            self._run_test_suite_async(questions)
        )


def create_retrieval_runner(
    embedder: Embeddings,
    embedded_doc_chunks: list[EmbeddedDocChunk],
    max_top_k: int,
    config: RunnerConfig
) -> RetrievalRunner:
    
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
    
    llm_model = get_llm(config.llm_model)
    
    judge = TwoWayChunkRelevanceJudge(
        llm_client=llm_model,
        relevance_judge_prompt_loader=TwoWayRelevanceJudgePromptLoader(
            prompt_config=config.relevance_judge_prompt_config
        )
    )
    
    return RetrievalRunner(
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        two_way_chunk_relevance_judge=judge
    )