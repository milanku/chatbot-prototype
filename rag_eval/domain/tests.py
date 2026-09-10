from pydantic import BaseModel

from rag_eval.domain.judgement import JudgedChunk
from rag_eval.domain.question import Question


class RetrievalTestCase(BaseModel):
    question: Question
    required_chunks: list[JudgedChunk]
    relevant_chunks: list[JudgedChunk]
    irrelevant_chunks: list[JudgedChunk]

class RetrievalTestSuite(BaseModel):
    test_cases: list[RetrievalTestCase]