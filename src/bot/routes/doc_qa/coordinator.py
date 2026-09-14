from dataclasses import dataclass

from bot.logging import log_event
from bot.models.doc_qa.references import DocReference
from bot.routes.doc_qa.chunk_relevance_judge import ChunkRelevanceJudge
from bot.routes.doc_qa.reranker import Reranker
from bot.routes.doc_qa.retriever import ChunksRetriever
from bot.routes.doc_qa.synthesizer import AnswerSynthesizer
from bot.routes.doc_qa.verifier.answer_verifier import AnswerVerifier


@dataclass
class DocsAnswerResult:
    answer_text: str
    references: list[DocReference]

class DocsAnswerCoordinator:
    def __init__(
        self,
        *,
        retriever: ChunksRetriever,
        synthesizer: AnswerSynthesizer,
        verifier: AnswerVerifier,
        judge: ChunkRelevanceJudge,
        reranker: Reranker,
    ):
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._verifier = verifier
        self._judge = judge
        self._reranker = reranker
    
    def answer(self, question: str) -> DocsAnswerResult:
        chunks = self._retriever.retrieve(question=question)
        reranked_chunks = self._reranker.rerank(question, chunks)
        chunks_judged_as_required = self._judge.judge_candidates_for_single_question(
            question=question,
            candidate_chunks=reranked_chunks
        )
        
        log_event(
            event="reranked_chunks",
            payload={
                "question": question,
                "reranked_chunks": [(hit.chunk.content, f"{hit.chunk.retrieval_score:.4f},{hit.chunk.reranker_score:.4f}") for hit in chunks_judged_as_required],
            }
        )
        
        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[hit.chunk.content for hit in chunks_judged_as_required],
        )
        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=[hit.chunk for hit in chunks_judged_as_required],
        )
        
        if(draft_verification.is_supported):
            return DocsAnswerResult(
                answer_text=draft_answer,
                references=[hit.chunk.doc_reference for hit in chunks_judged_as_required]
            )
        else:
            return DocsAnswerResult(
                answer_text="The answer could not be verified against the provided evidence.",
                references=[]
            )