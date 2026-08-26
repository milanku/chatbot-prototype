from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference
from bot.routes.doc_qa.retriever import DocHitsRetriever
from bot.routes.doc_qa.synthesizer import AnswerSynthesizer
from bot.routes.doc_qa.verifier.answer_verifier import AnswerVerifier


@dataclass
class DocsAnswer:
    answer_text: str
    references: list[DocReference]

class DocsAnswerCoordinator:
    def __init__(
        self,
        *,
        retriever: DocHitsRetriever,
        synthesizer: AnswerSynthesizer,
        verifier: AnswerVerifier,
    ):
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._verifier = verifier
    
    def answer(self, question: str) -> DocsAnswer:
        doc_hits = self._retriever.retrieve(question=question)
        draft_answer = self._synthesizer.synthesize(
            question=question,
            chunks_for_synthesis=[hit.content for hit in doc_hits],
        )
        draft_verification = self._verifier.verify_answer(
            question=question,
            draft_answer=draft_answer,
            source_evidence=doc_hits,
        )
        
        if(draft_verification.is_supported):
            return DocsAnswer(
                answer_text=draft_answer,
                references=[hit.doc_reference for hit in doc_hits]
            )
        else:
            return DocsAnswer(
                answer_text="The answer could not be verified against the provided evidence.",
                references=[]
            )