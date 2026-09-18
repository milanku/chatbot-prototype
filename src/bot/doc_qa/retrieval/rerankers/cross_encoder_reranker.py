from typing import cast

import numpy as np
from FlagEmbedding import FlagReranker  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from bot.config.reranker import LocalRerankerConfig
from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.rerankers.models import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, reranker: FlagReranker, config: LocalRerankerConfig):
        self._reranker = reranker
        self._config = config
    
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
    
    def rerank(self, query: str, documents: list[DocChunk]) -> list[DocChunk]:
        pairs = [(query, doc.content) for doc in documents]
        
        scores = self._compute_scores(pairs)
        
        for doc, score in zip(documents, scores):
            doc.reranker_score = float(score)
        
        return sorted(documents, key=lambda doc: doc.reranker_score or 0, reverse=True)[:self._config.top_k]


        
