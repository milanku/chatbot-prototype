from bot.doc_qa.retrieval.retrievers.factory import ChunksRetriever
from bot.doc_qa.indexing.store_models import DocChunk
from rag_eval.domain.question import (
    Question,
    QuestionCollection,
)
from rag_eval.domain.retrieval import CandidateChunks


def get_candidate_chunks(
    *,
    question: Question,
    common_doc_chunks: dict[str, DocChunk],
    retrievers: list[ChunksRetriever]
) -> CandidateChunks:
    candidate_answers_ids: set[str] = set()
    for retriever in retrievers:
        candidate_answers_ids.update(chunk.chunk_id for chunk in retriever.retrieve(question=question.content))
    
    candidate_chunks: list[DocChunk] = []
    for chunk_id in candidate_answers_ids:
        chunk = common_doc_chunks.get(chunk_id)
        if chunk is not None:
            candidate_chunks.append(chunk)

    return CandidateChunks(
        question_id=question.id,
        chunks=candidate_chunks,
    )

def get_candidate_chunks_for_question_collection(
    *,
    question_collection: QuestionCollection,
    common_doc_chunks: dict[str, DocChunk],
    retrievers: list[ChunksRetriever]
) -> list[CandidateChunks]:
    sets: list[CandidateChunks] = []
    
    for item in question_collection.questions:
        candidate_chunks = get_candidate_chunks(
            question=item,
            common_doc_chunks=common_doc_chunks,
            retrievers=retrievers
        )
        sets.append(candidate_chunks)
    
    return sets