from prompt_evals.claim_verifier.models import (
    ClaimVerifierBatchEvaluationResult,
    ClaimVerifierTestBatch,
    ClaimVerifierTestClaim,
    ClaimVerifierTestClaimResult,
)

from bot.doc_qa.verification.models import ExtractedClaim, VerifiedClaim
from bot.doc_qa.verification.verifier import ClaimsVerifier
from bot.llm.client import LLMClient
from bot.logging import log_event


class ClaimVerifierEvaluator:
    def __init__(self, llm_client: LLMClient, verifier: ClaimsVerifier):
        self.llm_client = llm_client
        self.verifier = verifier

    def _run_case_batch(
        self,
        batch: ClaimVerifierTestBatch,
    ) -> list[VerifiedClaim]:

        results = self.verifier.verify_against_evidence(
            user_query=batch.user_query,
            claims=[
                ExtractedClaim(
                    claim_id=case.claim_id,
                    claim=case.claim,
                    source_text=case.source_text,
                )
                for case in batch.claims
            ],
            evidence_chunks=batch.evidence_chunks,
        )

        return results

    def evaluate_claim_verifier_batch(
        self,
        batch: ClaimVerifierTestBatch,
    ) -> ClaimVerifierBatchEvaluationResult:

        verified_claims = self._run_case_batch(batch=batch)

        # Compare the expected verification status with the actual verification status and categorize the results
        passed: list[ClaimVerifierTestClaimResult] = []
        failed: list[ClaimVerifierTestClaimResult] = []

        for claim, verified in zip(batch.claims, verified_claims, strict=True):
            is_correct = (
                claim.expected_verification_status.value == verified.verification_status.value
            )
            result = ClaimVerifierTestClaimResult(
                verified_claim=ClaimVerifierTestClaim(
                    **verified.extracted_claim.model_dump(),
                    expected_verification_status=claim.expected_verification_status,
                ),
                evidence_chunks=verified.evidence_chunks,
                actual_verification_status=verified.verification_status,
                reason=verified.reason,
                correct=is_correct,
            )
            if is_correct:
                passed.append(result)
            else:
                failed.append(result)

        log_event(
            event="claim_verification.batch_evaluation",
            payload={
                "batch_id": batch.id,
                "failed_claims": [f.model_dump() for f in failed],
                "correct_claims": [c.model_dump() for c in passed],
                "all_correct": len(failed) == 0,
            },
        )

        return ClaimVerifierBatchEvaluationResult(
            batch_id=batch.id, correct_claims=passed, failed_claims=failed
        )
