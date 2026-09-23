from pydantic import BaseModel

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.judges.chunk_judge import JudgedChunk
from bot.doc_qa.retrieval.judges.models import ChunkRequirement


class ChunksRetrieval(BaseModel):
    retrieved_chunks: list[DocChunk]
    bm25_retrieved_chunks: list[DocChunk]
    reranked_chunks: list[DocChunk]
    chunks_judged_as_required: list[JudgedChunk[ChunkRequirement]]
    
QuestionChunksRetrieval = dict[str, ChunksRetrieval]