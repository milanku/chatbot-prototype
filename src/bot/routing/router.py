from bot.models.routing import Recipe, RouterDecision


def route(message: str) -> RouterDecision:
    text = message.strip().lower()

    # Primitive routing logic for now --- TODO: replace with a proper router

    # Explain how sum was computed, list transactions that contributed to it
    if any(keyphrase in text for keyphrase in ["list", "this sum"]):
        return RouterDecision(recipe=Recipe.TX_EXPLAIN, confidence=0.9)

    # Spending total
    if any(keyphrase in text for keyphrase in ["how much", "spent on"]):
        return RouterDecision(recipe=Recipe.TX_SUMMARY, confidence=0.9)
      
    # Answer questions about docs
    if any(
        keyphrase in text
        for keyphrase in ["how to", "what is", "explain", "where can i", "documentation"]
    ):
        return RouterDecision(recipe=Recipe.DOCS_ANSWER, confidence=0.9)

    # Out of scope
    return RouterDecision(recipe=Recipe.OUT_OF_SCOPE, confidence=0.9)