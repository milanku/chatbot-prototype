from dataclasses import dataclass

from bot.models.doc_qa.retrieval import DocHit
from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class DocSynthesizerPromptInput:
    question: str
    hits: list[DocHit]       
        
class DocAnswerSynthesizerPromptLoader(PromptLoader[DocSynthesizerPromptInput]):
    def build_user_prompt(self, input: DocSynthesizerPromptInput) -> str:
        hits = input.hits
        question = input.question
        
        context_block = "\n\n".join(hit.content for hit in hits)
        return (
            f"Answer the user's question based on the following relevant information retrieved from the documents:\n\n"
            f"Question: {question}\n\n"
            f"Retrieved information: \n{context_block}\n\n"
        )