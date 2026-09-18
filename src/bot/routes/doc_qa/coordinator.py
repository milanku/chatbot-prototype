from dataclasses import dataclass

from bot.judges.models import ChunkFilter
from bot.logging import log_event
from bot.models.doc_qa.chunks import DocChunk
from bot.models.doc_qa.references import DocReference
from bot.retrievers.models import ChunksRetriever
from bot.routes.doc_qa.reranker import Reranker
from bot.routes.doc_qa.synthesizer import AnswerSynthesizer
from bot.routes.doc_qa.verifier.answer_verifier import AnswerVerifier
from bot.routes.doc_qa.verifier.models import ClaimVerificationStatus


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
    
    def answer(self, question: str) -> DocsAnswerResult:
        # Agreggate chunks from all retrievers
        chunks: dict[str, DocChunk] = {}
        for retriever in self._retrievers:
            for chunk in retriever.retrieve(question=question):
                chunks[chunk.chunk_id] = chunk
                
        # Rerank all retrieved chunks
        reranked_chunks = self._reranker.rerank(question, list(chunks.values()))
        
        # Pick only chunks with correct tag according to the judge
        chunks_judged_as_required = self._judge.filter_chunks(
            question=question,
            candidate_chunks=reranked_chunks,
        )
        
        #log_event(
        #    event="required_chunks",
        #    payload={
        #        "question": question,
        #        "required_chunks": [(hit.chunk.content, f"{hit.chunk.retrieval_score:.4f},{hit.chunk.reranker_score:.4f}") for hit in chunks_judged_as_required],
        #    }
        #)
        
        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[hit.content for hit in chunks_judged_as_required],
        )
        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=[hit for hit in chunks_judged_as_required],
        )
        
        if(draft_verification.is_supported):
            return DocsAnswerResult(
                answer_text=draft_answer,
                references=[hit.doc_reference for hit in chunks_judged_as_required]
            )
        else:
            for claim in draft_verification.verified_claims:
                if claim.verification_status != ClaimVerificationStatus.SUPPORTED:
                    log_event(
                        event="UNVERIFIED_CLAIM",
                        payload={
                            "question": question,
                            "draft_answer": draft_answer,
                            "unverified_claim": claim,
                        }
                    )
            return DocsAnswerResult(
                answer_text="The answer could not be verified against the provided evidence.",
                references=[]
            )