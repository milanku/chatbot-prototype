from typing import cast

from transformers import AutoModel

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.rerankers.models import JinaRerankerModel, Reranker


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

    def rerank(self, query: str, documents: list[DocChunk]) -> list[DocChunk]:
        if not documents:
            return []

        contents = [doc.content for doc in documents]
        results = self._reranker.rerank(query, contents)

        return [documents[result["index"]] for result in results]