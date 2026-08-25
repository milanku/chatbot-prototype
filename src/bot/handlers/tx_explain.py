from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.query import (
    TXExplainQueryExtraction,
)
from bot.routes.tx_qa.explain_parser import parse_explain_query_from_user_message
from bot.routes.tx_qa.explain_parser_prompt_loader import TXExplainParserPromptLoader
from bot.trace_context import get_current_session_id


class TxExplainHandler(RouteHandler):
    
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        explain_query_parser_prompt_loader: TXExplainParserPromptLoader
    ) -> None:
        self._llm_client = llm_client
        self._explain_query_parser_prompt_loader = explain_query_parser_prompt_loader
    
    def handle(self, *, message: str, session_state: SessionState)  -> RouteHandlerResult:
        session_id = get_current_session_id()
        txs_results = session_state.txs_results
                
        query_extraction: TXExplainQueryExtraction | None = parse_explain_query_from_user_message(llm_client=self._llm_client, msg=message, prompt_loader=self._explain_query_parser_prompt_loader)
        
        if query_extraction is None:
            log_event(
                event="tx_explain.parse_query.error",
                payload={
                    "ok": False,
                    "session_id": session_id,
                    "reason": None,
                    "msg": "parse_explain_query_from_user_message returned None. Could not extract a transaction result reference from the message.",
                },
            )
            answer_text = "Sorry, your query could not be processed.\n"
            return RouteHandlerResult(
                answer_text=answer_text,
                new_state=session_state,
                references=[],
            )
        
        reference_offset = query_extraction.raw_query_data.reference_offset
        reference_count = query_extraction.raw_query_data.reference_count
        reason = query_extraction.reason

        if reference_offset is None or reference_count is None:
            log_event(
                event="tx_explain.parse_query.missing_reference",
                payload={
                    "ok": False,
                    "session_id": session_id,
                    "reference_offset": reference_offset,
                    "reference_count": reference_count,
                    "reason": reason,
                    "msg": "Could not extract a complete transaction result reference from the message.",
                },
            )
            answer_text = "Sorry, I could not tell which transaction summary you want me to explain. Please ask about the latest result or a specific previous result.\n"
            return RouteHandlerResult(
                answer_text=answer_text,
                new_state=session_state,
                references=[],
            )
        
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
                    
        return RouteHandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )
