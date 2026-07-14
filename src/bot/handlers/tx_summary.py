from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

from bot.handlers.models import HandlerResult
from bot.llm import client
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.repository import TransactionsRepository, TxFilter
from bot.models.tx_qa.results import TxQAQueryResult
from bot.routes.tx_qa import parse
from bot.routes.tx_qa.compute import compute_total_amount
from bot.routes.tx_qa.synthesize import synthesize_tx_summary


class TxSummaryHandler:
    def __init__(self, *, tx_repository: TransactionsRepository, llm_client: client.LLMClient):
        self._tx_repository = tx_repository
        self._llm_client = llm_client

    def handle(self, *, message:str, session_id:str, session_state: SessionState) -> HandlerResult:
        parsed_query = parse.parse_query(llm_client=self._llm_client, msg=message)
        new_state = session_state
    
        if parsed_query is None:
            log_event(
                event="tx_qa.parse_query.error",
                payload={
                    "message": "Parsed query is None. Could not extract label and date range from the message.",
                    "session_id": session_id,
                }
            )
            answer_text = "Sorry, I couldn't understand your query. Please make sure to include a label (food, pets, other) and a date range (e.g., 2026-01-01 - 2026-01-31).\n"
        else:
            log_event(
                event="tx_qa.parse_query.success",
                payload={
                    "label": parsed_query.label,
                    "start": parsed_query.start.isoformat(),
                    "end": parsed_query.end.isoformat(),
                    "session_id": session_id,
                }
            )
            tx_filter = TxFilter(
                label=parsed_query.label,
                start=parsed_query.start,
                end=parsed_query.end,
                direction="spend",  # For simplicity, we only consider spending transactions in this example
            )
            log_event(
                event="tx_qa.query",
                payload={
                    "label": parsed_query.label,
                    "start": parsed_query.start.isoformat(),
                    "end": parsed_query.end.isoformat(),
                    "direction": tx_filter.direction,
                    "session_id": session_id,
                }
            )
            
            txs = self._tx_repository.list_transactions(tx_filter)
            total_spent = compute_total_amount(txs)
            log_event(
                event="tx_qa.query_result",
                payload={
                    "total_spent": f"{total_spent:.2f}",
                    "num_transactions": len(txs),
                    "session_id": session_id,
                }
            )
            query_result: TxQAQueryResult = TxQAQueryResult(
                query=parsed_query,
                total=Decimal(total_spent), 
                created_at=datetime.now(
                    timezone.utc
                ),  # Using current UTC time as a timestamp
            )
            # Update state with the new query result
            new_state = replace(
                session_state,
                txs_results=session_state.txs_results + (query_result,),
            )
            answer_text = synthesize_tx_summary(parsed_query, float(total_spent))
        
        return HandlerResult(
            answer_text=answer_text,
            new_state=new_state,
            references=[],
        )