from pydantic import BaseModel

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.verification.models import ClaimVerificationResult, ExtractedClaim


class ClaimVerifierTestClaim(ExtractedClaim, BaseModel):
    expected_verification_status: ClaimVerificationResult


class ClaimVerifierTestBatch(BaseModel):
    id: str
    user_query: str
    claims: list[ClaimVerifierTestClaim]
    evidence_chunks: list[DocChunk]


class ClaimVerifierTestClaimResult(BaseModel):
    verified_claim: ClaimVerifierTestClaim
    evidence_chunks: list[DocChunk]
    reason: str | None = None
    actual_verification_status: ClaimVerificationResult
    correct: bool


class ClaimVerifierBatchEvaluationResult(BaseModel):
    batch_id: str
    correct_claims: list[ClaimVerifierTestClaimResult]
    failed_claims: list[ClaimVerifierTestClaimResult]
