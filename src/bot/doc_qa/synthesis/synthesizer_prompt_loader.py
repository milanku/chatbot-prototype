from dataclasses import dataclass

from bot.prompts.models import PromptLoader


@dataclass(frozen=True)
class SynthesizerPromptInput:
    question: str
    chunks: list[str]       

class SynthesizerPromptLoader(PromptLoader[SynthesizerPromptInput]):
    def build_user_prompt(self, input: SynthesizerPromptInput) -> str:
        chunks = input.chunks
        question = input.question
        
        context_block = "\n\n".join(chunks)
        return (
            f"Answer the user's question based on the following relevant information retrieved from the documents:\n\n"
            f"Question: {question}\n\n"
            f"Retrieved information:\n{context_block}\n\n"
        )