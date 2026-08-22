from bot.handlers.base import Handler
from bot.handlers.models import HandlerResult
from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.doc_qa.doc_repository import DocRepository
from bot.models.memory import SessionState
from bot.routes.doc_qa.synthesize import synthesize_doc_answer
from bot.routes.doc_qa.verifier.claim_extractor import (
    SentenceForExtraction,
    extract_claims_from_sentences,
)
from bot.routes.doc_qa.verifier.models import ClaimVerificationStatus
from bot.routes.doc_qa.verifier.verifier import verify_claims_against_evidence
from bot.routes.doc_qa.verify import filter_relevant_hits
from bot.trace_context import get_current_session_id


class DocsAnswerHandler(Handler):
    def __init__(self, doc_repository: DocRepository, llm_client: LLMClient):
        self._doc_repository = doc_repository
        self._llm_client = llm_client

    def handle(self, *, message:str, session_state: SessionState)  -> HandlerResult:
        session_id = get_current_session_id()
        top_k_chunks = self._doc_repository.get_top_k_chunks(message, top_k=5)
        log_event(
            event="doc_qa.retrieval",
            payload={
                "query": message,
                "retrieved_chunks": [{"id": hit.id, "score": hit.score, "content": hit.content} for hit in top_k_chunks],
                "session_id": session_id,
            }
        )
        filtered_hits = filter_relevant_hits(hits=top_k_chunks, absolute_relevance_threshold=0.4, relative_relevance_threshold=0.85)
        log_event(
            event="doc_qa.relevance_filter",
            payload={
                "query": message,
                "retrieved_chunks": len(top_k_chunks),
                "relevant_chunks": len(filtered_hits),
                "session_id": session_id,
            }
        )

        answer_text = synthesize_doc_answer(llm_client=self._llm_client, question=message, hits=filtered_hits)
        references = [hit.doc_reference for hit in filtered_hits]
        
        extracted_claims = extract_claims_from_sentences(
            llm_client=self._llm_client,
            sentences=[
                SentenceForExtraction(chunk_id=f"S{i:03d}", content=sentence.strip())
                for i, sentence in enumerate(answer_text.split("."))
            ]
        )
        
        log_event(
            event="doc_qa.claim_extraction",
            payload={
                "query": message,
                "extracted_claims": [claim.model_dump() for claim in extracted_claims],
                "session_id": session_id,
            }
        )
        
        verified_claims = verify_claims_against_evidence(
            llm_client=self._llm_client,
            user_query=message,
            claims=extracted_claims,
            evidence_chunks=filtered_hits
        )
        
        log_event(
            event="doc_qa.claim_verification",
            payload={
                "query": message,
                "verified_claims": [claim.model_dump() for claim in verified_claims],
                "session_id": session_id,
            }
        )
        
        if all(claim.verification_status == ClaimVerificationStatus.SUPPORTED for claim in verified_claims):
            # All claims are verified
            return HandlerResult(
                answer_text=answer_text,
                new_state=session_state,  # In this simple example, we don't update the session state
                references=references,
            )   
        else:
            # TODO: Some claims are not_supported or contradicted - retry?
            answer_text += "\n\nNote: Some claims in the answer could not be verified against the provided evidence."
            return HandlerResult(
                answer_text=answer_text,
                new_state=session_state,
                references=references,
            )