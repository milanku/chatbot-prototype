from bot.models.repository import DocHit


def filter_relevant_hits(*, hits: list[DocHit], relevance_threshold: float = 0.75) -> list[DocHit]:
    return [hit for hit in hits if hit.score >= relevance_threshold]