from typing import Any, Protocol

from bot.doc_qa.indexing.models import DocChunk


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[DocChunk]) -> list[DocChunk]: ...
    
class JinaRerankerModel(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[dict[str, Any]]:
        ...
        
    def eval(self) -> None:
        ...