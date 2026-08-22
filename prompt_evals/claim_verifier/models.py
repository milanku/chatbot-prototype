from pydantic import BaseModel

from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.verifier.models import (
    ClaimVerificationStatus,
    ExtractedClaim,
)


class ClaimVerifierTestClaim(ExtractedClaim, BaseModel):
    expected_verification_status: ClaimVerificationStatus
    
class ClaimVerifierTestBatch(BaseModel):
    id: str
    user_query: str
    claims: list[ClaimVerifierTestClaim]
    evidence_chunks: list[DocHit]
    
class ClaimVerifierTestClaimResult(BaseModel):
    verified_claim: ClaimVerifierTestClaim
    evidence_chunks: list[DocHit]
    reason: str | None = None
    actual_verification_status: ClaimVerificationStatus
    correct: bool

class ClaimVerifierBatchEvaluationResult(BaseModel):
    batch_id: str
    correct_claims: list[ClaimVerifierTestClaimResult]
    failed_claims: list[ClaimVerifierTestClaimResult]