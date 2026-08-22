from dataclasses import dataclass

from bot.models.prompts import PromptLoader
from bot.routes.doc_qa.verifier.models import SentenceForExtraction


@dataclass(frozen=True)
class ClaimExtractionPromptInput:
    sentences: list[SentenceForExtraction]
    
class ClaimExtractionPromptLoader(PromptLoader[ClaimExtractionPromptInput]):
    def build_user_prompt(self, input: ClaimExtractionPromptInput) -> str:
        chunks = "\n\n".join(f"Sentence ID: [{sentence.chunk_id}]\nContent: {sentence.content}" for sentence in input.sentences)
        
        return (
            "Extract the independently verifiable claims from the following sentences.\n\n"
            f"Sentences:\n\n{chunks}\n"
        )