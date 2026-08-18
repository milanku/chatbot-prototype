from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

from bot.handlers.base import TxBaseHandler
from bot.handlers.models import HandlerResult
from bot.models.memory import SessionState
from bot.models.tx_qa.query import TxQAQuery
from bot.models.tx_qa.repository import TxFilter
from bot.models.tx_qa.results import TxQAQueryResult
from bot.routes.tx_qa import parse
from bot.routes.tx_qa.compute import compute_total_amount
from bot.routes.tx_qa.synthesize import synthesize_tx_summary
from bot.routes.tx_qa.timeframe.resolver import resolve_date_range_from_raw_query


class TxSummaryHandler(TxBaseHandler):

    def handle(self, *, message: str, session_id: str, session_state: SessionState) -> HandlerResult:
        new_state = session_state
        parsed_raw_query = parse.parse_raw_tx_query_from_user_message(llm_client=self._llm_client, user_msg=message)
        
        raw_query_data = parsed_raw_query.raw_query_data
        # reason = parsed_raw_query.reason
        
        date_range = resolve_date_range_from_raw_query(raw_query_data, today=date.today())
        if date_range is None:
            return HandlerResult(
                answer_text="Sorry, I could not determine the date range for your query. Please make sure to specify a valid timeframe (e.g., 'last month', 'from January 1st to January 31st').\n",
                new_state=new_state,
                references=[],
            )   
        start_date, end_date = date_range
        query = TxQAQuery(
            label=raw_query_data.label,
            start_date=start_date,
            end_date=end_date,
            direction=raw_query_data.direction,
        )
                    
        tx_filter = TxFilter(
            label=query.label,
            start_date=query.start_date,
            end_date=query.end_date,
            direction=query.direction,
        )
        
        txs = self._tx_repository.list_transactions(tx_filter)
        total_amount = compute_total_amount(txs)
        query_result: TxQAQueryResult = TxQAQueryResult(
            query=query,
            transactions=txs,
            total=Decimal(total_amount), 
            created_at=datetime.now(
                timezone.utc
            ),  # Using current UTC time as a timestamp
        )
        # Update state with the new query result
        new_state = replace(
            session_state,
            txs_results=session_state.txs_results + (query_result,),
        )
        answer_text = synthesize_tx_summary(query, float(total_amount))
        
        return HandlerResult(
            answer_text=answer_text,
            new_state=new_state,
            references=[],
        )