from bot.llm.client import LLMClient
from bot.routes.doc_qa.verifier.claim_extraction_prompt_loader import (
    ClaimExtractionPromptInput,
    ClaimExtractionPromptLoader,
)
from bot.routes.doc_qa.verifier.models import (
    ExtractedClaim,
    ExtractionResult,
    SentenceForExtraction,
)


def extract_claims_from_sentences(
    *, 
    llm_client: LLMClient,
    sentences: list[SentenceForExtraction],
    claim_extraction_prompt_loader: ClaimExtractionPromptLoader
) -> list[ExtractedClaim]:
    """
    Extracts claims from sentences.

    Args:
        llm_client (LLMClient): The LLM client to use for claim extraction.
        sentences (list[SentenceForExtraction]): The sentences from which to extract claims.

    Returns:
        list[ExtractedClaim]: A list of extracted claims.
    """

    system_prompt = claim_extraction_prompt_loader.load_system_instructions()
    user_prompt = claim_extraction_prompt_loader.build_user_prompt(ClaimExtractionPromptInput(sentences=sentences))
    
    llm_structured_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=ExtractionResult,
        system_instructions=system_prompt,
    )
    
    return [
        ExtractedClaim(
            claim_id=f"CLAIM_{i:03d}",
            claim=claim.claim,
            source_sentence_ids=claim.source_sentence_ids,
            source_text=claim.source_text
        )
        for i, claim in enumerate(llm_structured_response.claims)
    ] if llm_structured_response and llm_structured_response.claims else []