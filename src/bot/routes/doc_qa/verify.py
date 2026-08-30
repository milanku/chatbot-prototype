from bot.models.doc_qa.retrieval import DocHit


def filter_relevant_hits(*, hits: list[DocHit], absolute_relevance_threshold: float = 0.75, relative_relevance_threshold: float = 0.85) -> list[DocHit]:
    max_score = max((hit.retrieval_score for hit in hits), default=0)
    return [hit for hit in hits if hit.retrieval_score >= absolute_relevance_threshold and hit.retrieval_score >= relative_relevance_threshold * max_score]