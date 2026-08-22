from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.verifier.models import (
    ClaimVerificationResult,
    ExtractedClaim,
    VerifiedClaim,
    VerifiedClaimsLLMOutputFormat,
)
from bot.routes.doc_qa.verifier.verifier_prompt_loader import (
    ClaimVerifierPromptInput,
    PromptLoader,
)


def merge_claims_with_verification_results(
    *,
    claims: list[ExtractedClaim],
    verification_results: list[ClaimVerificationResult],
    evidence_chunks: list[DocHit]
) -> list[VerifiedClaim]:
    """
    Merges extracted claims with their corresponding verification results.

    Args:
        claims (list[ExtractedClaim]): The list of extracted claims.
        verification_results (list[ClaimVerificationResult]): The list of claim verification results.
        evidence_chunks (list[DocHit]): The list of evidence chunks.

    Returns:
        list[VerifiedClaim]: A list of verified claims with their verification status and evidence.
    """
    claim_dict = {claim.claim_id: claim for claim in claims}
    
    verified_claims: list[VerifiedClaim] = []
    for result in verification_results:
        claim = claim_dict.get(result.claim_id)
        if claim:
            verified_claims.append(
                VerifiedClaim(
                    extracted_claim=claim,
                    evidence_chunks=evidence_chunks,
                    verification_status=result.verification_status,
                    reason=result.reason
                )
            )
    
    return verified_claims
    
def verify_claims_against_evidence(
    llm_client: LLMClient,
    *,
    user_query: str,
    claims: list[ExtractedClaim],
    evidence_chunks: list[DocHit],
    verification_prompt_loader: PromptLoader[ClaimVerifierPromptInput],
) -> list[VerifiedClaim]:
    """
    Verifies claims against evidence chunks.

    Args:
        llm_client (LLMClient): The LLM client to use for claim verification.
        user_query (str): The user's original query.
        claims (list[ExtractedClaim]): The claims to verify.
        evidence_chunks (list[DocHit]): The evidence chunks to verify against.

    Returns:
        list[VerifiedClaim]: A list of verified claims.
    """
    
    system_prompt = verification_prompt_loader.load_system_instructions()
    user_prompt = verification_prompt_loader.build_user_prompt(ClaimVerifierPromptInput(
        user_query=user_query,
        retrieved_evidence_chunks=evidence_chunks,
        claims=claims
    ))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=VerifiedClaimsLLMOutputFormat,
        system_instructions=system_prompt,
    )
    
    merged_verification_results = merge_claims_with_verification_results(
        claims=claims,
        verification_results=llm_structured_response.verified_claims,
        evidence_chunks=evidence_chunks
    ) if llm_structured_response and llm_structured_response.verified_claims else []
    
    log_event(
        event="claim_verification.verification_results",
        payload={
            "merged_verification_results": [result.model_dump() for result in merged_verification_results],
        }   
    )
    
    return merged_verification_results
    