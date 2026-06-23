from bot.handlers.models import HandlerResult
from bot.llm import llm_client
from bot.models.memory import SessionState
from bot.models.tx_qa.query import TxFilter
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.tx_qa import parse


class TxListHandler:
    def __init__(self, tx_repository: TransactionsRepository, llm_client: llm_client.LLMClient):
        self._tx_repository = tx_repository
        self._llm_client = llm_client
    
    def handle(self, *, message:str, session_id:str, session_state: SessionState, trace)  -> HandlerResult:
        parsed_query = parse.parse_query(session_id=session_id, llm_client=self._llm_client, msg=message)
        
        if parsed_query is None:
            answer_text = "Sorry, I couldn't understand your query. Please make sure to include a label (food, pets, other) and a date range (e.g., 2026-01-01 - 2026-01-31).\n"
            trace(
                "tx_qa.parse_query",
                ok=False,
            )
        else:
            tx_filter = TxFilter(
                label=parsed_query.label,
                start=parsed_query.start,
                end=parsed_query.end,
                direction="spend",  # For simplicity, we only consider spending transactions in this example
            )
            trace(
                "tx_qa.parse_query.result",
                label=parsed_query.label,
                start=parsed_query.start.isoformat(),
                end=parsed_query.end.isoformat(),
                direction=tx_filter.direction,
            )
            
            txs = self._tx_repository.list_transactions(tx_filter)
            
            trace(
                "tx_qa.query_result",
                num_transactions=len(txs),
            )
            
            if not txs:
                answer_text = "No transactions found for the specified query.\n"
            else:
                answer_text = "Here are your transactions:\n"
                for tx in txs:
                    answer_text += f"- {tx.date}: {tx.amount:.2f} EUR to {tx.other_account} ({tx.description})\n"
                    
        return HandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )