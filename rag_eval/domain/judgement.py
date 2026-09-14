import enum

from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk


class ChunkRelevance(str, enum.Enum):
    REQUIRED="REQUIRED"
    RELEVANT="RELEVANT"
    IRRELEVANT="IRRELEVANT"

class JudgedChunk(BaseModel):
    chunk: DocChunk
    relevance: ChunkRelevance
    reason: str

class QuestionChunkJudgments(BaseModel):
    question_id: str
    chunk_judgements: list[JudgedChunk]

class CandidateRetrievalJudgments(BaseModel):
    question_chunk_judgements: list[QuestionChunkJudgments]
    
class QuestionQuality(str, enum.Enum):
    ACCEPT="ACCEPT"
    REJECT="REJECT"

class JudgedQuestion(BaseModel):
    question_id: str
    quality: QuestionQuality
    reason: str

class QuestionJudgments(BaseModel):
    question_judgements: list[JudgedQuestion]