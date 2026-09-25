from bot.doc_qa.verification.claim_extractor_prompt_loader import (
    ClaimExtractorPromptInput,
    ClaimExtractorPromptLoader,
)
from bot.doc_qa.verification.models import (
    ExtractedClaim,
    ExtractionResult,
    SentenceForExtraction,
)
from bot.llm.client import LLMClient


class ClaimExtractor:
    def __init__(
        self, *, llm_client: LLMClient, claim_extraction_prompt_loader: ClaimExtractorPromptLoader
    ):
        self._claim_extraction_prompt_loader = claim_extraction_prompt_loader
        self._llm_client = llm_client

    def _split_text_into_sentences(
        self,
        *,
        text: str,
    ) -> list[SentenceForExtraction]:
        sentences = text.split(".")
        return [
            SentenceForExtraction(chunk_id=f"CHUNK_{i:03d}", content=sentence.strip())
            for i, sentence in enumerate(sentences)
            if sentence.strip()
        ]

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
        sentences = self._split_text_into_sentences(text=text)
        user_prompt = self._claim_extraction_prompt_loader.build_user_prompt(
            ClaimExtractorPromptInput(sentences=sentences)
        )

        llm_structured_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=ExtractionResult,
            system_instructions=system_prompt,
        )

        return [
            ExtractedClaim(
                **claim.model_dump(),
                claim_id=f"CLAIM_{i:03d}",
            )
            for i, claim in enumerate(llm_structured_response.claims)
        ]
