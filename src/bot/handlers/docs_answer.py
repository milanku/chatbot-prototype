from bot.handlers.models import HandlerResult
from bot.llm import llm_client
from bot.models.doc_qa.docs import DocRepository
from bot.models.memory import SessionState
from bot.routes.doc_qa.synthesize import synthesize_doc_answer
from bot.routes.doc_qa.verify import filter_relevant_hits


class DocsAnswerHandler:
    def __init__(self, doc_repository: DocRepository, llm_client: llm_client.LLMClient):
        self._doc_repository = doc_repository
        self._llm_client = llm_client

    def handle(self, *, message:str, session_id:str, session_state: SessionState, trace)  -> HandlerResult:
        top_k_chunks = self._doc_repository.get_top_k_chunks(message, top_k=5)
        trace(
            "doc_qa.retrieval",
            query=message,
            retrieved_chunks=[{"id": hit.id, "score": hit.score, "content": hit.content} for hit in top_k_chunks]
        )
        filtered_hits = filter_relevant_hits(hits=top_k_chunks, absolute_relevance_threshold=0.5, relative_relevance_threshold=0.85)
        trace(
            "doc_qa.relevance_filter",
            query=message,
            retrieved_chunks=len(top_k_chunks),
            relevant_chunks=len(filtered_hits),
        )
        answer_text = synthesize_doc_answer(llm_client=self._llm_client, question=message, hits=filtered_hits)
        references = [hit.doc_reference for hit in filtered_hits]
        
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,  # In this simple example, we don't update the session state
            references=references,
        )
