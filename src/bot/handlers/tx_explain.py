from bot.handlers.base import TxBaseHandler
from bot.handlers.models import HandlerResult
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.query import (
    TXExplainQueryExtraction,
)
from bot.models.tx_qa.results import TxQAQueryResult
from bot.routes.tx_qa.explain_parser import parse_explain_query_from_user_message
from bot.trace_context import get_current_session_id


class TxExplainHandler(TxBaseHandler):
        
    def handle(self, *, message: str, session_state: SessionState)  -> HandlerResult:
        session_id = get_current_session_id()
        txs_results = session_state.txs_results
                
        query_extraction: TXExplainQueryExtraction | None = parse_explain_query_from_user_message(llm_client=self._llm_client, msg=message)
        
        # TODO: Fix the logic here to handle missing extraction, reason.
        if query_extraction is None:
            log_event(
                event="tx_explain.parse_query.error",
                payload={
                    "ok": False,
                    "session_id": session_id,
                    "msg": "parse_explain_query_from_user_message returned None. Could not extract label and date range from the message.",
                },
            )
            answer_text = "Sorry, your query could not be processed.\n"
            return HandlerResult(
                answer_text=answer_text,
                new_state=session_state,
                references=[],
            )
        
        reference_count = query_extraction.raw_query_data.reference_count
        reference_offset = query_extraction.raw_query_data.reference_offset
        # reason = query_extraction.reason
        
        related_query_results: list[TxQAQueryResult] = []
        
        if reference_offset is not None and reference_count is not None:
            related_query_results = list(
                txs_results[max(0, len(txs_results) - reference_offset - reference_count):len(txs_results) - reference_offset]
            )

        if not related_query_results:
            answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
        else:
            answer_text = "Here are the transactions that contributed to your selected sum:\n"
            for tx_result in related_query_results:
                for tx in tx_result.transactions:
                    answer_text += f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"
                    
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )