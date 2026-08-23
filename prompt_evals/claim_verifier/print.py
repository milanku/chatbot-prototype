from pathlib import Path
from pprint import pformat
from typing import TypedDict

from bot.config.prompts_config import PromptConfig
from prompt_evals.claim_verifier.models import ClaimVerifierBatchEvaluationResult


class FailedClaimSummary(TypedDict):
    batch_id: str
    claim_id: str | None
    claim: str
    correct: bool
    expected_verdict: str
    actual_verdict: str


def print_results(
    prompt_config: PromptConfig,
    batch_results: list[ClaimVerifierBatchEvaluationResult],
) -> None:
    all_claims = [
        claim
        for batch in batch_results
        for claim in batch.correct_claims + batch.failed_claims
    ]

    total_claims_count = len(all_claims)
    correct_claims_count = sum(
        claim.correct
        for claim in all_claims
    )

    total_batches_count = len(batch_results)
    correct_batches_count = sum(
        len(batch.failed_claims) == 0
        for batch in batch_results
    )

    claim_accuracy = (
        correct_claims_count / total_claims_count * 100 if total_claims_count > 0 else 0.0
    )

    failed_claims: list[FailedClaimSummary] = [
        {
            "batch_id": batch.batch_id,
            "claim_id": claim.verified_claim.claim_id,
            "claim": claim.verified_claim.claim,
            "correct": claim.correct,
            "expected_verdict": claim.verified_claim.expected_verification_status.value,
            "actual_verdict": claim.actual_verification_status.value,
        }
        for batch in batch_results
        for claim in batch.failed_claims
    ]
    
    print_template = Path("prompt_evals/claim_verifier/evaluation_print_template.txt").read_text(
        encoding="utf-8"
    )
    
    print(
        print_template.format(
            prompt_version=prompt_config.version,
            total_batches_count=total_batches_count,
            passed_batches_count=correct_batches_count,
            
            total_claims_count=total_claims_count,
            correct_claims_count=correct_claims_count,
            incorrect_claims_count=total_claims_count - correct_claims_count,
            claim_accuracy=f"{claim_accuracy:.1f}%",
            failed_claims=pformat(failed_claims, sort_dicts=False),
        )
    )
