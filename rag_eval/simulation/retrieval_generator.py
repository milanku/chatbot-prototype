from rag_eval.questions.models import Question
from rag_eval.simulation.models import ChunksRetrieval, QuestionChunksRetrieval

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.judges.chunk_requirement_judge import ChunkRequirementJudge
from bot.doc_qa.retrieval.judges.models import (
    ChunkRequirement,
)
from bot.doc_qa.retrieval.rerankers.cross_encoder_reranker import (
    Reranker,
)
from bot.doc_qa.retrieval.retrievers.factory import (
    ChunksRetriever,
)


class RetrievalGenerator:
    def __init__(
        self,
        *,
        embeddings_retriever: ChunksRetriever,
        bm25_retriever: ChunksRetriever,
        reranker: Reranker,
        two_way_chunk_relevance_judge: ChunkRequirementJudge,
    ):
        self._embeddings_retriever = embeddings_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._two_way_chunk_relevance_judge = two_way_chunk_relevance_judge

    def precompute_retrieval_sets_for_questions(
        self,
        questions: list[Question],
    ) -> QuestionChunksRetrieval:
        q_retrievals: QuestionChunksRetrieval = {}
        for question in questions:
            embedding_retrieved_chunks = self._embeddings_retriever.retrieve(
                question=question.content
            )
            bm25_retrieved_chunks = self._bm25_retriever.retrieve(question=question.content)

            retrieved_chunks: list[DocChunk] = []
            retrieved_chunks.extend(embedding_retrieved_chunks)
            retrieved_chunks.extend(bm25_retrieved_chunks)

            # Combine the retrieved doc chunks from both retrievers before reranking and remove duplicates
            retrieved_chunks = list({chunk.chunk_id: chunk for chunk in retrieved_chunks}.values())
            reranked_chunks = self._reranker.rerank(question.content, retrieved_chunks)

            judged_doc_chunks_all = self._two_way_chunk_relevance_judge.judge_chunks(
                question=question.content, candidate_chunks=reranked_chunks
            )

            chunks_judged_as_required = [
                chunk
                for chunk in judged_doc_chunks_all
                if chunk.relevance == ChunkRequirement.REQUIRED
            ]

            q_retrievals[question.id] = ChunksRetrieval(
                retrieved_chunks=embedding_retrieved_chunks,
                bm25_retrieved_chunks=bm25_retrieved_chunks,
                reranked_chunks=reranked_chunks,
                chunks_judged_as_required=chunks_judged_as_required,
            )

        return q_retrievals
