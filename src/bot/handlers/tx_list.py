from datetime import date

from bot.handlers.models import RouteHandler, RouteHandlerResult
from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.memory import SessionState
from bot.models.tx_qa.repository import TransactionsRepository, TxFilter
from bot.routes.tx_qa.parse import parse_raw_tx_query_from_user_message
from bot.routes.tx_qa.timeframe.resolver import resolve_date_range_from_raw_query
from bot.routes.tx_qa.timeframe_parser_prompt_loader import TimeframeParserPromptLoader
from bot.trace_context import get_current_session_id


class TxListHandler(RouteHandler):
    
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tx_repository: TransactionsRepository,
        timeframe_parser_prompt_loader: TimeframeParserPromptLoader
    ) -> None:
        self._llm_client = llm_client
        self._tx_repository = tx_repository
        self._timeframe_parser_prompt_loader = timeframe_parser_prompt_loader
    
    def handle(self, *, message:str, session_state: SessionState) -> RouteHandlerResult:
        session_id = get_current_session_id()
        parsed_tx_query = parse_raw_tx_query_from_user_message(llm_client=self._llm_client, user_msg=message, prompt_loader=self._timeframe_parser_prompt_loader)
        
        raw_query_data = parsed_tx_query.raw_query_data
        date_range = resolve_date_range_from_raw_query(raw_query_data, today=date.today())
        
        if(date_range is None):
            return RouteHandlerResult(
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
                
        return RouteHandlerResult(
            answer_text=answer_text,
            new_state=session_state,
            references=[],
        )