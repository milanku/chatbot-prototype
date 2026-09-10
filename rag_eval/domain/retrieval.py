from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk


class CandidateChunks(BaseModel):
    question_id: str
    chunks: list[DocChunk]
    
class CandidateRetrieval(BaseModel):
    candidate_chunks: list[CandidateChunks]