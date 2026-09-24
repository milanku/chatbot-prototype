from typing import Any, Protocol, cast

from transformers import AutoModel

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.rerankers.models import Reranker


# Protocol for Jina reranker from huggingface
class JinaRerankerModel(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[str],
    ) -> list[dict[str, Any]]:
        ...
        
    def eval(self) -> None:
        ...
        
class JinaReranker(Reranker):
    def __init__(self):
        self._reranker = cast(
            JinaRerankerModel,
            AutoModel.from_pretrained(
                pretrained_model_name_or_path="jinaai/jina-reranker-v3.5",
                dtype="auto",
                trust_remote_code=True,
            ),
        )
        self._reranker.eval()

    def rerank(self, query: str, chunks: list[DocChunk]) -> list[DocChunk]:
        if not chunks:
            return []

        contents = [
            doc.content
            for doc in chunks
        ]
        
        results = self._reranker.rerank(query, contents)

        return [
            chunks[result["index"]]
            for result in results
        ]