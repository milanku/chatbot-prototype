from dataclasses import dataclass
from pathlib import Path

from bot.routes.doc_qa.verifier.models import SentenceForExtraction


@dataclass(frozen=True)
class ClaimExtractionPromptInput:
    sentences: list[SentenceForExtraction]
    
def load_claim_extraction_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_claim_extraction_system_prompt(template: str)-> str:
    return template

def build_claim_extraction_user_prompt(input: ClaimExtractionPromptInput) -> str:
    chunks = "\n\n".join(f"Sentence ID: [{sentence.chunk_id}]\nContent: {sentence.content}" for sentence in input.sentences)
    return (
        "Extract the independently verifiable claims from the following sentences.\n\n"
        f"Sentences:\n\n{chunks}\n"
    )