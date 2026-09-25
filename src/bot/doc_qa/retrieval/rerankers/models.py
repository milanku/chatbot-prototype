from typing import Protocol

from bot.doc_qa.indexing.models import DocChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[DocChunk]) -> list[DocChunk]: ...
