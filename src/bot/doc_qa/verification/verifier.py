from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.verification.models import (
    ClaimVerificationResult,
    ExtractedClaim,
    VerifiedClaim,
    VerifiedClaimsLLMOutputFormat,
)
from bot.doc_qa.verification.verifier_prompt_loader import (
    ClaimVerifierPromptInput,
    ClaimVerifierPromptLoader,
)
from bot.llm.client import LLMClient
from bot.logging import log_event

DEFAULT_RETRIES = 3

class ClaimsVerifier:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        claim_verifier_prompt_loader: ClaimVerifierPromptLoader
    ):
        self._llm_client = llm_client
        self._claim_verifier_prompt_loader = claim_verifier_prompt_loader
    
    def _merge_claims_with_verification_results(
        self,
        *,
        claims: list[ExtractedClaim],
        verification_results: list[ClaimVerificationResult],
        evidence_chunks: list[DocChunk]
    ) -> list[VerifiedClaim]:
        """
        Merges extracted claims with their corresponding verification results.

        Args:
            claims (list[ExtractedClaim]): The list of extracted claims.
            verification_results (list[ClaimVerificationResult]): The list of claim verification results.
            evidence_chunks (list[DocChunk]): The list of evidence chunks.

        Returns:
            list[VerifiedClaim]: A list of verified claims with their verification status and evidence.
        """
        claim_dict = {
            claim.claim_id: claim
            for claim in claims
        }
        
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
    
    def _check_output_validity(
        self,
        claims: list[ExtractedClaim],
        verification_results: list[ClaimVerificationResult]
    ) -> bool:
        """
        Checks the validity of the LLM output: Ensures that each claim has a corresponding verification result.

        Args:
            claims (list[ExtractedClaim]): The list of extracted claims to be verified against the evidence chunks.
            verification_results (list[ClaimVerificationResult]): The list of claim verification results.

        Returns:
            bool: True if the output is valid, False otherwise.
        """
        
        expected_ids = {
            claim.claim_id
            for claim in claims
        }
        llm_output_ids = {
            result.claim_id
            for result in verification_results
        }
        
        if len(claims) != len(verification_results) or expected_ids != llm_output_ids:
            log_event(
                event="claim_verification.invalid_output",
                payload={
                    "expected_ids": expected_ids,
                    "output_ids": llm_output_ids,
                    "claims_count": len(claims),
                    "verification_results_count": len(verification_results),
                }
            )
            return False
        
        return True
    
    def verify_against_evidence(
        self,
        *,
        user_query: str,
        claims: list[ExtractedClaim],
        evidence_chunks: list[DocChunk],
    ) -> list[VerifiedClaim]:
        """
        Verifies claims against evidence chunks.

        Args:
            user_query (str): The user's original query.
            claims (list[ExtractedClaim]): The claims to verify.
            evidence_chunks (list[DocChunk]): The evidence chunks to verify against.

        Returns:
            list[VerifiedClaim]: A list of verified claims.
        """
        
        system_prompt = self._claim_verifier_prompt_loader.load_system_instructions()
        user_prompt = self._claim_verifier_prompt_loader.build_user_prompt(
            ClaimVerifierPromptInput(
                user_query=user_query,
                retrieved_evidence_chunks=evidence_chunks,
                claims=claims
            )
        )
        
        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=VerifiedClaimsLLMOutputFormat,
            system_instructions=system_prompt,
            retries=DEFAULT_RETRIES,
            check_is_output_valid=lambda output: self._check_output_validity(
                claims=claims,
                verification_results=output.verified_claims
            )
        )
        
        merged_verification_results = self._merge_claims_with_verification_results(
            claims=claims,
            verification_results=llm_structured_response.verified_claims,
            evidence_chunks=evidence_chunks
        ) if llm_structured_response and llm_structured_response.verified_claims else []
        
        log_event(
            event="claim_verification.verification_results",
            payload={
                "merged_verification_results": [
                    result.model_dump()
                    for result in merged_verification_results
                ],
            }   
        )
        
        return merged_verification_results
    