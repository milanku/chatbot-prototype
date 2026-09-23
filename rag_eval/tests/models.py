from pydantic import BaseModel

from bot.doc_qa.retrieval.judges.models import ChunkRelevance, JudgedChunk
from rag_eval.questions.models import Question


class RetrievalTestCase(BaseModel):
    question: Question
    required_chunks: list[JudgedChunk[ChunkRelevance]]
    relevant_chunks: list[JudgedChunk[ChunkRelevance]]
    irrelevant_chunks: list[JudgedChunk[ChunkRelevance]]