from dataclasses import dataclass
from pathlib import Path

from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.verifier.claim_extractor import ExtractedClaim


@dataclass(frozen=True)
class ClaimVerifierPromptInput:
    user_query: str
    retrieved_evidence_chunks: list[DocHit]
    claims: list[ExtractedClaim]
    
def load_verifier_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_verifier_system_prompt(template: str)-> str:
    return template

def build_verifier_user_prompt(input: ClaimVerifierPromptInput) -> str:
    evidence_segment = "\n\n".join(f"Evidence ID: [{chunk.id}]\nContent: {chunk.content}" for chunk in input.retrieved_evidence_chunks)
    claims_segment = "\n\n".join(f"Claim ID: [{claim.claim_id}]\nContent: {claim.claim}" for claim in input.claims)

    #"\n\n".join(f"Sentence ID: [{sentence.chunk_id}]\nContent: {sentence.content}" for sentence in input.sentences)
    
    return (
        "Verify the following claims based on the retrieved evidence chunks.\n\n"
        f"USER QUERY:{input.user_query!r}\n"
        f"CLAIMS:{claims_segment}\n"
        f"RETRIEVED EVIDENCE CHUNKS:{evidence_segment}\n"
    )