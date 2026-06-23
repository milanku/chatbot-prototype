from bot.handlers.models import HandlerResult
from bot.llm import llm_client
from bot.models.memory import SessionState
from bot.models.repository import TransactionsRepository, TxFilter
from bot.routes.tx_qa.explain import TXExplainParseIntermediateResult, parse_explain_query


class TxExplain:
    def __init__(self, tx_respository: TransactionsRepository, llm_client: llm_client.LLMClient):
        self._tx_repository = tx_respository
        self._llm_client = llm_client
        
    def handle(self, *, message:str, session_id:str, session_state: SessionState, trace)  -> HandlerResult:
        txs_results = session_state.txs_results
                
        reference_result:TXExplainParseIntermediateResult = parse_explain_query(session_id=session_id, llm_client=self._llm_client, msg=message)
        
        trace(
            "tx_explain.parse_query",
            txs_results=len(txs_results),
            reference_offset=reference_result.reference_offset,
            reference_count=reference_result.reference_count,
            confidence=reference_result.confidence,
            reason=reference_result.reason,
        )
        
        related_results = []
        if reference_result.reference_offset is not None and reference_result.reference_count is not None:
            offset = reference_result.reference_offset
            count = reference_result.reference_count
            related_results = txs_results[max(0, len(txs_results) - offset - count):len(txs_results) - offset]
            trace(
                "tx_explain.find_reference",
                found=bool(related_results),
                reference_offset=reference_result.reference_offset,
                reference_count=reference_result.reference_count,
                confidence=reference_result.confidence,
                reason=reference_result.reason,
            )
        else:
            trace(
                "tx_explain.find_reference",
                found=False,
                confidence=reference_result.confidence,
                reason=reference_result.reason,
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