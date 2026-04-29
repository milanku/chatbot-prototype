def score_chunks_jaccard (input_tokens: list[str], doc_tokens: list[str]) -> float:
    # Simple Jaccard similarity
    input_set = set(input_tokens)
    doc_set = set(doc_tokens)
    intersection = input_set.intersection(doc_set)
    union = input_set.union(doc_set)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def vectors_cosine_similarity (a_vector: list[float], b_vector: list[float]) -> float:
    # Cosine similarity
    dot_product = sum(i * d for i, d in zip(a_vector, b_vector, strict=True))
    a_magnitude = sum(i ** 2 for i in a_vector) ** 0.5
    b_magnitude = sum(d ** 2 for d in b_vector) ** 0.5
    if a_magnitude == 0 or b_magnitude == 0:
        return 0.0
    return dot_product / (a_magnitude * b_magnitude)