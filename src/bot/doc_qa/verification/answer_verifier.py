from dataclasses import dataclass

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.verification.claim_extractor import (
    ClaimExtractor,
)
from bot.doc_qa.verification.models import (
    ClaimVerificationStatus,
    VerifiedClaim,
)
from bot.doc_qa.verification.verifier import ClaimsVerifier
from bot.llm.client import LLMClient


@dataclass
class AnswerVerification:
    is_supported: bool
    verified_claims: list[VerifiedClaim]

class AnswerVerifier:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        claim_extractor: ClaimExtractor,
        claim_verifier: ClaimsVerifier
    ):
        self._llm_client = llm_client
        self._claim_extractor = claim_extractor
        self._claim_verifier = claim_verifier
        
    def verify_answer(
        self,
        *,
        question: str,
        draft_answer: str,
        source_evidence: list[DocChunk],
    ) -> AnswerVerification:
        extracted_claims_from_draft_answer = self._claim_extractor.extract_claims_from_text(
            text=draft_answer
        )
        verified_claims = self._claim_verifier.verify_against_evidence(
            user_query=question,
            claims=extracted_claims_from_draft_answer,
            evidence_chunks=source_evidence,
        )
    
        return AnswerVerification(
            is_supported=all(
                claim.verification_status == ClaimVerificationStatus.SUPPORTED
                for claim in verified_claims
            ),
            verified_claims=verified_claims,
        )