from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.models import DocsAnswerResult, DocsAnswerStatus, InsufficientEvidenceReason
from bot.doc_qa.retrieval.judges.models import ChunkFilter
from bot.doc_qa.retrieval.rerankers.models import Reranker
from bot.doc_qa.retrieval.retrievers.models import ChunksRetriever
from bot.doc_qa.synthesis.synthesizer import AnswerSynthesizer
from bot.doc_qa.verification.answer_verifier import AnswerVerifier
from bot.doc_qa.verification.models import AnswerVerification, ClaimVerificationStatus
from bot.logging import LogLevel, log_event


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
        # FUTURE: When using multiple retrievers, score gets overridden by the last retriever that returns the chunk.
        # TODO: Merge scores from retrievers - consider adding a dict[retriever -> (score, rank)] to DocChunk model
        # Note: This is not a problem when using a single retriever + bm25.
        for retriever in self._retrievers:
            for chunk in retriever.retrieve(question=question):
                chunks[chunk.chunk_id] = chunk
        return list(chunks.values())

    def _insufficient_evidence_result(
        self,
        question: str,
        reason: InsufficientEvidenceReason,
    ) -> DocsAnswerResult:
        log_event(
            event="doc_qa.coordinator.insufficient_evidence",
            payload={
                "question": question,
                "reason": reason,
            },
            log_level=LogLevel.INFO,
        )
        return DocsAnswerResult(
            status=DocsAnswerStatus.INSUFFICIENT_EVIDENCE,
            answer_text="I couldn't find enough information in the documentation to answer your question.",
            references=[],
        )

    def _handle_verification_result(
        self,
        draft_verification: AnswerVerification,
        question: str,
        draft_answer: str,
        required_chunks: list[DocChunk],
    ) -> DocsAnswerResult:
        if draft_verification.is_supported:
            return DocsAnswerResult(
                status=DocsAnswerStatus.ANSWERED,
                answer_text=draft_answer,
                references=[hit.doc_reference for hit in required_chunks],
            )

        for claim in draft_verification.verified_claims:
            if claim.verification_status != ClaimVerificationStatus.SUPPORTED:
                log_event(
                    event="doc_qa.coordinator.error_unverified_claim",
                    payload={
                        "question": question,
                        "draft_answer": draft_answer,
                        "unverified_claim": claim,
                    },
                    log_level=LogLevel.WARNING,
                )

        return DocsAnswerResult(
            status=DocsAnswerStatus.DRAFT_VERIFICATION_FAIL,
            answer_text="I am not able to provide unambiguous answer to your question based on the available documentation.",
            references=[],
        )

    def answer(self, question: str) -> DocsAnswerResult:
        retrieved_chunks = self._retrieve(question)

        if not retrieved_chunks:
            return self._insufficient_evidence_result(
                question, InsufficientEvidenceReason.EMPTY_RETRIEVAL
            )

        reranked_chunks = self._reranker.rerank(question, retrieved_chunks)

        if not reranked_chunks:
            return self._insufficient_evidence_result(
                question, InsufficientEvidenceReason.EMPTY_RERANK
            )

        required_chunks = self._judge.filter_chunks(
            question=question,
            candidate_chunks=reranked_chunks,
        )

        if not required_chunks:
            return self._insufficient_evidence_result(
                question, InsufficientEvidenceReason.EMPTY_REQUIRED
            )

        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[hit.content for hit in required_chunks],
        )

        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=[hit for hit in required_chunks],
        )

        return self._handle_verification_result(
            draft_verification=draft_verification,
            question=question,
            draft_answer=draft_answer,
            required_chunks=required_chunks,
        )
