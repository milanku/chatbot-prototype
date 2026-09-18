from typing import Protocol

from bot.models.doc_qa.chunks import DocChunk


class Chunker(Protocol):    
    def split(self, text: str, file_name: str) -> list[DocChunk]:
        ...