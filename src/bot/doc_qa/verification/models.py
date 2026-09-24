from enum import Enum

from pydantic import BaseModel

from bot.doc_qa.indexing.models import DocChunk


class LLMExtractedClaim(BaseModel):
    claim: str
    source_sentence_ids: list[str] | None = None
    source_text: str | None = None

class ExtractionResult(BaseModel):
    claims: list[LLMExtractedClaim]
    
class ExtractedClaim(LLMExtractedClaim):
    claim_id: str | None = None
    
class SentenceForExtraction(BaseModel):
    chunk_id: str
    content: str
    
class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"

class ClaimVerificationResult(BaseModel):
    claim_id: str
    verification_status: ClaimVerificationStatus
    reason: str
    
class VerifiedClaim(BaseModel):
    extracted_claim: ExtractedClaim
    evidence_chunks: list[DocChunk]
    verification_status: ClaimVerificationStatus
    reason: str
    
class VerifiedClaimsLLMOutputFormat(BaseModel):
    verified_claims: list[ClaimVerificationResult]