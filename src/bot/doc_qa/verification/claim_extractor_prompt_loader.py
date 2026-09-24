from dataclasses import dataclass

from bot.doc_qa.verification.models import SentenceForExtraction
from bot.prompts.models import PromptLoader


@dataclass(frozen=True)
class ClaimExtractorPromptInput:
    sentences: list[SentenceForExtraction]
    
class ClaimExtractorPromptLoader(PromptLoader[ClaimExtractorPromptInput]):
    def build_user_prompt(self, input: ClaimExtractorPromptInput) -> str:
        chunks = "\n\n".join(
            f"Sentence ID: [{sentence.chunk_id}]\n"
            f"Content: {sentence.content}"
            for sentence in input.sentences
        )
        
        return (
            "Extract the independently verifiable claims from the following sentences.\n\n"
            f"Sentences:\n\n{chunks}\n"
        )