from dataclasses import dataclass

from bot.models.doc_qa.retrieval import DocHit
from bot.models.prompts import PromptLoader
from bot.routes.doc_qa.verifier.claim_extractor import ExtractedClaim


@dataclass(frozen=True)
class ClaimVerifierPromptInput:
    user_query: str
    retrieved_evidence_chunks: list[DocHit]
    claims: list[ExtractedClaim]
    
class ClaimVerifierPromptLoader(PromptLoader[ClaimVerifierPromptInput]):
    def build_user_prompt(self, input: ClaimVerifierPromptInput) -> str:
        claims_segment = "\n\n".join(f"Claim ID: [{claim.claim_id}]\nContent: {claim.claim}" for claim in input.claims)
        evidence_segment = "\n\n".join(f"Evidence ID: [{chunk.id}]\nContent: {chunk.content}" for chunk in input.retrieved_evidence_chunks)

        return (
            "Verify the following claims based on the retrieved evidence chunks.\n\n"
            f"USER QUERY:{input.user_query!r}\n"
            f"CLAIMS:{claims_segment}\n"
            f"RETRIEVED EVIDENCE CHUNKS:{evidence_segment}\n"
        )
