def score_chunks (input_tokens: list[str], doc_tokens: list[str]) -> float:
    # Simple Jaccard similarity
    input_set = set(input_tokens)
    doc_set = set(doc_tokens)
    intersection = input_set.intersection(doc_set)
    union = input_set.union(doc_set)
    if not union:
        return 0.0
    return len(intersection) / len(union)