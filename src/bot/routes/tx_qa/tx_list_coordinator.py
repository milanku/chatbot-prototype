from dataclasses import dataclass
from datetime import date

from bot.models.tx_qa.query import TxQuery
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.tx_qa.timeframe.resolve import resolve_date_range
from bot.routes.tx_qa.timeframe_parser import TimeframeParser
from bot.routes.tx_qa.tx_list_formatter import format_transactions


@dataclass
class TxListAnswer:
    answer: str
    
class TxListCoordinator:
    def __init__(self, *, tx_repository: TransactionsRepository, timeframe_parser: TimeframeParser):
        self._tx_repository = tx_repository
        self._timeframe_parser = timeframe_parser
    
    def answer(self, *, question: str, today: date) -> TxListAnswer:
        extraction = self._timeframe_parser.parse(question)
        date_range = resolve_date_range(extraction.timeframe, today)
        
        if(date_range is None):
            return TxListAnswer(
                answer="Sorry, I could not determine the date range for your query. Please make sure to specify a valid timeframe (e.g., 'last month', 'from January 1st to January 31st').\n",
            )
        
        start_date, end_date = date_range
        tx_query = TxQuery(
            label=extraction.timeframe.label,
            start_date=start_date,
            end_date=end_date,
            direction=extraction.timeframe.direction,
        )
        txs = self._tx_repository.list_transactions(tx_query)
        
        return TxListAnswer(answer=format_transactions(txs))