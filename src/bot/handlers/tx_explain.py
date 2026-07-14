from bot.handlers.base import TxBaseHandler
from bot.handlers.models import HandlerResult
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.query import TXExplainParseIntermediateResult
from bot.models.tx_qa.repository import TxFilter
from bot.models.tx_qa.results import TxQAQueryResult
from bot.routes.tx_qa.explain import parse_explain_query


class TxExplainHandler(TxBaseHandler):
        
    def handle(self, *, message:str, session_id:str, session_state: SessionState)  -> HandlerResult:
        txs_results = session_state.txs_results
                
        reference_result:TXExplainParseIntermediateResult | None = parse_explain_query(llm_client=self._llm_client, msg=message)
        
        # TODO: Fix the logic here to handle the case where reference_result is None.
        if reference_result is None:
            log_event(
                event="tx_explain.parse_query.error",
                payload={
                    "ok": False,
                    "session_id": session_id,
                },
            )
            answer_text = "Sorry, your query could not be processed.\n"
            return HandlerResult(
                answer_text=answer_text,
                new_state=session_state,
                references=[],
            )
        
        log_event(
            event="tx_explain.parse_query.success",
            payload={
                "txs_results": len(txs_results),
                "reference_offset": reference_result.reference_offset,
                "reference_count": reference_result.reference_count,
                "confidence": reference_result.confidence,
                "reason": reference_result.reason,
                "session_id": session_id,
            },
        )
        
        related_results: list[TxQAQueryResult] = []
        if reference_result.reference_offset is not None and reference_result.reference_count is not None:
            offset = reference_result.reference_offset
            count = reference_result.reference_count
            related_results = list(
                txs_results[max(0, len(txs_results) - offset - count):len(txs_results) - offset]
            )
            log_event(
                event="tx_explain.find_reference",
                payload={
                    "found": bool(related_results),
                    "reference_offset": reference_result.reference_offset,
                    "reference_count": reference_result.reference_count,
                    "confidence": reference_result.confidence,
                    "reason": reference_result.reason,
                    "session_id": session_id,
                },
            )
        else:
            log_event(
                event="tx_explain.find_reference",
                payload={
                    "found": False,
                    "confidence": reference_result.confidence,
                    "reason": reference_result.reason,
                    "session_id": session_id,
                },
            )

        if not related_results:
            answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
        else:
            answer_text = "Here are the transactions that contributed to your selected sum:\n"
            for tx_result in related_results:
                txs = self._tx_repository.list_transactions(
                    TxFilter(
                        label=tx_result.query.label,
                        start=tx_result.query.start,
                        end=tx_result.query.end,
                        direction="spend",
                    )
                )
                for tx in txs:
                    answer_text += f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"
                    
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )