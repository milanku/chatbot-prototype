import enum

from pydantic import BaseModel

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.judges.models import ChunkRelevance, JudgedChunk


class Question(BaseModel):
    id: str
    content: str
    reference_answer: str
    
class QuestionCollection(BaseModel):
    question_collection_id: str
    questions: list[Question]
    
QuestionCandidatesDict = dict[str, list[DocChunk]]

QuestionChunkJudgmentsMap = dict[str, list[JudgedChunk[ChunkRelevance]]]
    
class QuestionQuality(str, enum.Enum):
    ACCEPT="ACCEPT"
    REJECT="REJECT"

class JudgedQuestion(BaseModel):
    question_id: str
    quality: QuestionQuality
    reason: str