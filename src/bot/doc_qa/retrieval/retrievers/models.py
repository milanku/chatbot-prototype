from typing import Protocol

from bot.doc_qa.indexing.models import DocChunk


class ChunksRetriever(Protocol):
    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]: ...
