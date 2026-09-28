from pydantic import BaseModel
from rag_eval.questions.models import Question

from bot.doc_qa.retrieval.judges.models import ChunkRelevance, JudgedChunk


class RetrievalTestCase(BaseModel):
    question: Question
    required_chunks: list[JudgedChunk[ChunkRelevance]]
    relevant_chunks: list[JudgedChunk[ChunkRelevance]]
    irrelevant_chunks: list[JudgedChunk[ChunkRelevance]]
