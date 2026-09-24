from dataclasses import dataclass

from bot.doc_qa.indexing.models import DocChunk, DocReference
from bot.doc_qa.retrieval.judges.models import ChunkFilter
from bot.doc_qa.retrieval.rerankers.cross_encoder_reranker import Reranker
from bot.doc_qa.retrieval.retrievers.models import ChunksRetriever
from bot.doc_qa.synthesis.synthesizer import AnswerSynthesizer
from bot.doc_qa.verification.answer_verifier import AnswerVerification, AnswerVerifier
from bot.doc_qa.verification.models import ClaimVerificationStatus
from bot.logging import log_event


@dataclass
class DocsAnswerResult:
    answer_text: str
    references: list[DocReference]

class DocsAnswerCoordinator:
    def __init__(
        self,
        *,
        retrievers: list[ChunksRetriever],
        synthesizer: AnswerSynthesizer,
        verifier: AnswerVerifier,
        judge: ChunkFilter,
        reranker: Reranker,
    ):
        self._retrievers = retrievers
        self._synthesizer = synthesizer
        self._verifier = verifier
        self._judge = judge
        self._reranker = reranker
        
    def _retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        chunks: dict[str, DocChunk] = {}
        for retriever in self._retrievers:
            for chunk in retriever.retrieve(question=question):
                chunks[chunk.chunk_id] = chunk
        return list(chunks.values())
    
    def _insufficient_evidence_result(
        self,
        question: str,
    ) -> DocsAnswerResult:
        log_event(
            event="doc_qa.coordinator.insufficient_evidence",
            payload={"question": question},
        )
        return DocsAnswerResult(
            answer_text="I couldn't find enough information in the documentation to answer your question.",
            references=[]
        )
        
    def _handle_verification_result(
        self,
        draft_verification: AnswerVerification,
        question: str,
        draft_answer: str,
        required_chunks: list[DocChunk],
    ) -> DocsAnswerResult:
        if(draft_verification.is_supported):
            return DocsAnswerResult(
                answer_text=draft_answer,
                references=[hit.doc_reference for hit in required_chunks]
            )
        
        for claim in draft_verification.verified_claims:
            if claim.verification_status != ClaimVerificationStatus.SUPPORTED:
                log_event(
                    event="doc_qa.coordinator.verification_failure(unverified_claim)",
                    payload={
                        "question": question,
                        "draft_answer": draft_answer,
                        "unverified_claim": claim,
                    }
                )
        return DocsAnswerResult(
            answer_text="I am not able to provide unambiguous answer to your question based on the available documentation.",
            references=[]
        )
    
    def answer(self, question: str) -> DocsAnswerResult:
        retrieved_chunks = self._retrieve(question)
        
        reranked_chunks = self._reranker.rerank(
            question,
            retrieved_chunks
        )
        
        required_chunks = self._judge.filter_chunks(
            question=question,
            candidate_chunks=reranked_chunks,
        )
        
        if not required_chunks:
            return self._insufficient_evidence_result(question)
        
        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[
                hit.content
                for hit in required_chunks
            ],
        )
        
        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=[
                hit
                for hit in required_chunks
            ],
        )
        
        return self._handle_verification_result(
            draft_verification=draft_verification,
            question=question,
            draft_answer=draft_answer,
            required_chunks=required_chunks,
        )