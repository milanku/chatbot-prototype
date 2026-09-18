from dataclasses import dataclass

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.verification.models import ExtractedClaim
from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class ClaimVerifierPromptInput:
    user_query: str
    retrieved_evidence_chunks: list[DocChunk]
    claims: list[ExtractedClaim]
    
class ClaimVerifierPromptLoader(PromptLoader[ClaimVerifierPromptInput]):
    def build_user_prompt(self, input: ClaimVerifierPromptInput) -> str:
        claims_segment = "\n\n".join(f"Claim ID: [{claim.claim_id}]\nContent: {claim.claim}" for claim in input.claims)
        evidence_segment = "\n\n".join(f"Evidence ID: [{chunk.chunk_id}]\nContent: {chunk.content}" for chunk in input.retrieved_evidence_chunks)

        return (
            "Verify the following claims based on the retrieved evidence chunks.\n\n"
            f"USER QUERY:{input.user_query!r}\n"
            f"CLAIMS:{claims_segment}\n"
            f"RETRIEVED EVIDENCE CHUNKS:{evidence_segment}\n"
        )
