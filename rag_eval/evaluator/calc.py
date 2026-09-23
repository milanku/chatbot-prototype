def calc_recall(expected: set[str], retrieved: set[str]) -> float | None:
    return len(expected & retrieved) / len(expected) if expected else None

def calc_precision(expected: set[str], retrieved: set[str]) -> float | None:
    return len(expected & retrieved) / len(retrieved) if retrieved else None