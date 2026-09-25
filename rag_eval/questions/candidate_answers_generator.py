from rag_eval.questions.models import (
    Question,
    QuestionCandidatesDict,
    QuestionCollection,
)

from bot.doc_qa.indexing.store_models import DocChunk
from bot.doc_qa.retrieval.retrievers.factory import ChunksRetriever


def retrieve_candidate_chunks_union_for_question(
    *, question: Question, retrievers: list[ChunksRetriever]
) -> list[DocChunk]:
    """Using all retrievers creates a union of candidate chunks which
    could be used to generate answer (based by retriever selection).

    Args:
        question (Question): The question for which candidate chunks are being retrieved.
        retrievers (list[ChunksRetriever]): A list of retrievers used to fetch candidate chunks.

    Returns:
        list[DocChunk]: A list of candidate DocChunk objects retrieved for the question.
    """

    candidate_chunks: dict[str, DocChunk] = {}
    for retriever in retrievers:
        for chunk in retriever.retrieve(question=question.content):
            candidate_chunks[chunk.chunk_id] = chunk

    return list(candidate_chunks.values())


def retrieve_candidate_chunks_for_question_collection(
    *, question_collection: QuestionCollection, retrievers: list[ChunksRetriever]
) -> QuestionCandidatesDict:
    question_candidates: dict[str, list[DocChunk]] = {}

    for question in question_collection.questions:
        candidate_chunks = retrieve_candidate_chunks_union_for_question(
            question=question, retrievers=retrievers
        )
        question_candidates[question.id] = candidate_chunks

    return question_candidates
