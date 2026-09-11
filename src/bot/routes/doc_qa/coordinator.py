from dataclasses import dataclass

from bot.logging import log_event
from bot.models.doc_qa.references import DocReference
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
        reranker: Reranker
    ):
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._verifier = verifier
        self._reranker = reranker
    
    def answer(self, question: str) -> DocsAnswerResult:
        doc_hits = self._retriever.retrieve(question=question)
        reranked_doc_hits = self._reranker.rerank(question, doc_hits)
        
        
        log_event(
            event="reranked_doc_hits",
            payload={
                "question": question,
                "reranked_doc_hits": [(hit.content, f"{hit.retrieval_score:.4f},{hit.reranker_score:.4f}") for hit in reranked_doc_hits],
            }
        )
        
        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[hit.content for hit in reranked_doc_hits],
        )
        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=reranked_doc_hits,
        )
        
        if(draft_verification.is_supported):
            return DocsAnswerResult(
                answer_text=draft_answer,
                references=[hit.doc_reference for hit in reranked_doc_hits]
            )
        else:
            return DocsAnswerResult(
                answer_text="The answer could not be verified against the provided evidence.",
                references=[]
            )