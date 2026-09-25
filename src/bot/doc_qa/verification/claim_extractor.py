from bot.doc_qa.verification.claim_extractor_prompt_loader import (
    ClaimExtractorPromptInput,
    ClaimExtractorPromptLoader,
)
from bot.doc_qa.verification.models import (
    ClaimExtractionLLMOutput,
    ExtractedClaim,
)
from bot.llm.client import LLMClient
from bot.logging import LogLevel, log_event

DEFAULT_RETRIES = 3


class ClaimExtractor:
    def __init__(
        self, *, llm_client: LLMClient, claim_extraction_prompt_loader: ClaimExtractorPromptLoader
    ):
        self._claim_extraction_prompt_loader = claim_extraction_prompt_loader
        self._llm_client = llm_client

    def _check_is_output_valid(
        self,
        *,
        source_text: str,
        llm_output: ClaimExtractionLLMOutput,
    ) -> bool:
        """Checks if each extracted claim in the llm_output has source_span in the source_text verbatim

        Args:
            source_text (str): The original text from which claims were extracted.
            llm_output (ClaimExtractionLLMOutput): The output from the LLM containing extracted claims.

        Returns:
            bool: True if all extracted claims have valid source spans in the source text, False otherwise.
        """

        for extracted_claim in llm_output.claims:
            for source_span in extracted_claim.source_spans:
                if source_span not in source_text:
                    log_event(
                        event="doc_qa.claim_extractor.invalid_output",
                        log_level=LogLevel.ERROR,
                        payload={
                            "reason": "Source span not found in the source text.",
                            "source_text": source_text,
                            "extracted_claim": extracted_claim.claim,
                            "invalid_source_span": source_span,
                        },
                    )
                    return False

        return True

    def extract_claims_from_text(
        self,
        *,
        text: str,
    ) -> list[ExtractedClaim]:
        """
        Extracts claims from given text.

        Args:
            text (str): The text from which to extract claims.

        Returns:
            list[ExtractedClaim]: A list of extracted claims.
        """

        system_prompt = self._claim_extraction_prompt_loader.load_system_instructions()
        user_prompt = self._claim_extraction_prompt_loader.build_user_prompt(
            ClaimExtractorPromptInput(text=text)
        )

        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=ClaimExtractionLLMOutput,
            system_instructions=system_prompt,
            retries=DEFAULT_RETRIES,
            check_is_output_valid=lambda output: self._check_is_output_valid(
                source_text=text,
                llm_output=output,
            ),
        )

        return [
            ExtractedClaim(
                **claim.model_dump(),
                claim_id=f"CLAIM_{i:03d}",
            )
            for i, claim in enumerate(llm_structured_response.claims)
        ]
