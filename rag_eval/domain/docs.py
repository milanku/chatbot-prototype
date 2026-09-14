from pydantic import BaseModel

from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk


class DocChunks(BaseModel):
    chunks: list[DocChunk]
    
class EmbeddedDocChunks(BaseModel):
    embedded_chunks: list[EmbeddedDocChunk]