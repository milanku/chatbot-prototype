from typing import Protocol, cast

import numpy as np
from FlagEmbedding import FlagReranker  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from transformers import AutoModel, PreTrainedModel

from bot.models.doc_qa.retrieval import DocHit


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[DocHit]) -> list[DocHit]: ...
    
class CrossEncoderReranker(Reranker):
    def __init__(self, reranker: FlagReranker):
        self._reranker = reranker
    
    def _compute_scores(
        self,
        pairs: list[tuple[str, str]],
    ) -> NDArray[np.float64]:
        scores = cast(
            NDArray[np.float64] | None,
            self._reranker.compute_score(pairs), #pyright: ignore[reportUnknownMemberType]
        )

        if scores is None:
            raise RuntimeError("Reranker failed to compute scores")

        return np.asarray(scores, dtype=np.float64)
    
    def rerank(self, query: str, documents: list[DocHit]) -> list[DocHit]:
        pairs = [(query, doc.content) for doc in documents]
        
        scores = self._compute_scores(pairs)
        
        for doc, score in zip(documents, scores):
            doc.reranker_score = float(score)
        
        return sorted(documents, key=lambda doc: doc.reranker_score or 0, reverse=True)

class JinaReranker(Reranker):
    def __init__(self):
        self._reranker = cast(
            PreTrainedModel,
            AutoModel.from_pretrained(
                "jinaai/jina-reranker-v3.5",
                dtype="auto",
                trust_remote_code=True,
            ),
        )
        self._reranker.eval()

    def rerank(self, query: str, documents: list[DocHit]) -> list[DocHit]:
        if not documents:
            return []

        contents = [doc.content for doc in documents]
        results = self._reranker.rerank(query, contents)

        return [documents[result["index"]] for result in results]