from typing import Protocol

from bot.models.doc_qa.chunks import DocChunk


class ChunksRetriever(Protocol):
    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        ...