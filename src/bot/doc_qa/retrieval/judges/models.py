from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from bot.doc_qa.indexing.models import DocChunk

T = TypeVar("T", bound=StrEnum)

class ChunkRequirement(StrEnum):
    REQUIRED = "REQUIRED"
    NOT_REQUIRED = "NOT_REQUIRED"
    
class ChunkRelevance(StrEnum):
    REQUIRED = "REQUIRED"
    RELEVANT = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"

class ChunkJudgement(BaseModel, Generic[T]):
    chunk_id: str
    relevance: T
    reason: str

class JudgeOutputFormat(BaseModel, Generic[T]):
    results: list[ChunkJudgement[T]]
    
class JudgedChunk(BaseModel, Generic[T]):
    chunk: DocChunk
    relevance: T
    reason: str
    
# Generate_structured_output can't depend on generics
class ChunkRequirementJudgeOutputFormat(
    JudgeOutputFormat[ChunkRequirement]
):
    pass


class ChunkRelevanceJudgeOutputFormat(
    JudgeOutputFormat[ChunkRelevance]
):
    pass
    
class ChunkFilter(Protocol):
    def filter_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[DocChunk]:
        ...