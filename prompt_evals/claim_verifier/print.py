from bot.config.prompts_config import PromptConfig
from prompt_evals.claim_verifier.models import ClaimVerifierBatchEvaluationResult


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
        all(f.correct for f in batch.failed_claims)
        for batch in batch_results
    )

    print()
    print("=" * 60)
    print("CLAIM VERIFIER EVALUATION")
    print("=" * 60)

    print(f"Prompt:              {prompt_config.version}")
    print(f"Batches:             {total_batches_count}")
    print(f"Completely correct:  {correct_batches_count}/{total_batches_count}")
    print(f"Batch accuracy:      {correct_batches_count / total_batches_count:.1%}")
    print()
    print(f"Claims:              {total_claims_count}")
    print(f"Correct:             {correct_claims_count}")
    print(f"Incorrect:           {total_claims_count - correct_claims_count}")
    print(f"Claim accuracy:      {correct_claims_count / total_claims_count:.1%}")

    print()
    print("FAILURES")
    print("-" * 60)

    for batch in batch_results:
        for claim in batch.correct_claims + batch.failed_claims:
            if not claim.correct:
                print(
                    f"{batch.batch_id} / {claim}: "
                    f"expected={claim.verified_claim.expected_verification_status}, "
                    f"actual={claim.actual_verification_status}"
                )

                if claim.verified_claim:
                    print(f"  reason: {claim.reason}")