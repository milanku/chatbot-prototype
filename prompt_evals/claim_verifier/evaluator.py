from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.doc_qa.verifier.models import ExtractedClaim, VerifiedClaim
from bot.doc_qa.verifier.verifier import verify_claims_against_evidence
from bot.doc_qa.verifier.verifier_prompt_loader import (
    ClaimVerifierPromptLoader,
)
from prompt_evals.claim_verifier.models import (
    ClaimVerifierBatchEvaluationResult,
    ClaimVerifierTestBatch,
    ClaimVerifierTestClaim,
    ClaimVerifierTestClaimResult,
)


def _run_case_batch(
    llm_client: LLMClient,
    *,
    batch: ClaimVerifierTestBatch,
    prompt_loader: ClaimVerifierPromptLoader,
) -> list[VerifiedClaim]:
        
    results = verify_claims_against_evidence(
        llm_client=llm_client,
        user_query=batch.user_query,
        claims=[    
                ExtractedClaim(
                    claim_id=case.claim_id,
                    claim=case.claim,
                    source_text=case.source_text,
                    source_sentence_ids=case.source_sentence_ids
                )
                for case in batch.claims
        ],
        evidence_chunks=batch.evidence_chunks,
        verification_prompt_loader=prompt_loader
    )
    
    return results

def evaluate_claim_verifier_batch(
    llm_client: LLMClient,
    *,
    batch: ClaimVerifierTestBatch,
    prompt_loader: ClaimVerifierPromptLoader,
) -> ClaimVerifierBatchEvaluationResult:
    
    verified_claims = _run_case_batch(
        llm_client=llm_client,
        batch=batch,
        prompt_loader=prompt_loader
    )
    
    # Compare the expected verification status with the actual verification status and categorize the results
    correct: list[ClaimVerifierTestClaimResult] = []
    failed: list[ClaimVerifierTestClaimResult] = []
    for claim, verified in zip(batch.claims, verified_claims):
        is_correct = claim.expected_verification_status == verified.verification_status.value
        result = ClaimVerifierTestClaimResult(
            verified_claim=ClaimVerifierTestClaim(
                **verified.extracted_claim.model_dump(),
                expected_verification_status=claim.expected_verification_status
            ),
            evidence_chunks=verified.evidence_chunks,
            actual_verification_status=verified.verification_status,
            reason=verified.reason,
            correct=is_correct
        )
        if is_correct:
            correct.append(result)
        else:
            failed.append(result)
            
    log_event(
        event="claim_verification.batch_evaluation",
        payload={
            "batch_id": batch.id,
            "failed_claims": [f.model_dump() for f in failed],
            "correct_claims": [c.model_dump() for c in correct],
            "all_correct": len(failed) == 0,
        }
    )
    
    return ClaimVerifierBatchEvaluationResult(
        batch_id=batch.id,
        correct_claims=correct,
        failed_claims=failed
    )