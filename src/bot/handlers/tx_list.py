from datetime import date

from bot.handlers.base import TxBaseHandler
from bot.handlers.models import HandlerResult
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.repository import TxFilter
from bot.routes.tx_qa import parse
from bot.routes.tx_qa.timeframe.resolver import resolve_date_range_from_raw_query


class TxListHandler(TxBaseHandler):
    
    def handle(self, *, message:str, session_id:str, session_state: SessionState) -> HandlerResult:
        parsed_tx_query = parse.parse_raw_tx_query_from_user_message(llm_client=self._llm_client, user_msg=message)
        
        raw_query_data = parsed_tx_query.raw_query_data
        date_range = resolve_date_range_from_raw_query(raw_query_data, today=date.today())
        
        if(date_range is None):
            return HandlerResult(
                answer_text="Sorry, I could not determine the date range for your query. Please make sure to specify a valid timeframe (e.g., 'last month', 'from January 1st to January 31st').\n",
                new_state=session_state,
                references=[],
            )
        
        start_date, end_date = date_range
        
        log_event(
            event="tx_qa.parse_tx_query.success",
            payload={
                "label": raw_query_data.label,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "direction": raw_query_data.direction,
                "session_id": session_id,
            },
        )
        
        tx_filter = TxFilter(
            label=raw_query_data.label,
            start_date=start_date,
            end_date=end_date,
            direction=raw_query_data.direction,
        )
        txs = self._tx_repository.list_transactions(tx_filter)
        
        log_event(
            event="tx_qa.query_result",
            payload={
                "num_transactions": len(txs),
                "session_id": session_id,
            },
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