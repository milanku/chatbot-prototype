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
            self._reranker.compute_score(pairs),  # pyright: ignore[reportUnknownMemberType]
        )

        if scores is None:
            raise RuntimeError("Reranker failed to compute scores")

        return np.asarray(scores, dtype=np.float64)

    def rerank(self, query: str, chunks: list[DocChunk]) -> list[DocChunk]:
        query_chunk_pairs = [(query, doc.content) for doc in chunks]

        scores = self._compute_scores(query_chunk_pairs)

        scored_docs = [
            doc.model_copy(update={"reranker_score": float(score)})
            for doc, score in zip(chunks, scores, strict=True)
        ]

        return sorted(
            scored_docs,
            key=lambda doc: doc.reranker_score if doc.reranker_score is not None else float("-inf"),
            reverse=True,
        )[: self._config.top_k]
