from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk

T = TypeVar("T", bound=StrEnum)

class TwoWayChunkRequirement(StrEnum):
    REQUIRED = "REQUIRED"
    NOT_REQUIRED = "NOT_REQUIRED"
    
class ThreeWayChunkRelevance(StrEnum):
    REQUIRED = "REQUIRED"
    NOT_REQUIRED = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"

class JudgedChunkOutput(BaseModel, Generic[T]):
    chunk_id: str
    relevance: T
    reason: str

class JudgeOutputFormat(BaseModel, Generic[T]):
    results: list[JudgedChunkOutput[T]]
    
    
# Generate_structured_output can't depend on generics
class TwoWayJudgeOutputFormat(
    JudgeOutputFormat[TwoWayChunkRequirement]
):
    pass


class ThreeWayJudgeOutputFormat(
    JudgeOutputFormat[ThreeWayChunkRelevance]
):
    pass
    
class ChunkFilter(Protocol):
    def filter_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[DocChunk]:
        ...