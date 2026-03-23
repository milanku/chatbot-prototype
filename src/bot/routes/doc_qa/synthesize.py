from bot.models.repository import DocHit


def synthesize_doc_answer(question: str, hits: list[DocHit]) -> str:
    # For simplicity, we'll just concatenate the retrieved documents.
    # In a real implementation, you might want to use a language model to generate a more coherent answer.
    if not hits:
        return "I'm sorry, I couldn't find any relevant information in the documents."

    docs_summary = "\n\n".join(hit.content for hit in hits)
    return f"Based on the documents I found, here is the information related to your question:\n\n{docs_summary}"