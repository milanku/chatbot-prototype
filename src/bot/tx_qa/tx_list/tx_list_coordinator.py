from dataclasses import dataclass
from datetime import date

from bot.models.tx_qa.query import TxQuery
from bot.models.tx_qa.repository import TransactionsRepository
from bot.tx_qa.timeframe.resolve import resolve_date_range
from bot.tx_qa.parsing.timeframe_parser import TimeframeParser
from bot.tx_qa.tx_list.formatter import format_transactions


@dataclass
class TxListResult:
    answer: str
    
class TxListCoordinator:
    def __init__(self, *, tx_repository: TransactionsRepository, timeframe_parser: TimeframeParser):
        self._tx_repository = tx_repository
        self._timeframe_parser = timeframe_parser
    
    def answer(self, *, question: str, today: date) -> TxListResult:
        extraction = self._timeframe_parser.parse(question)
        date_range = resolve_date_range(extraction.timeframe, today)
        
        if(date_range is None):
            return TxListResult(
                answer="Sorry, I could not determine the date range for your query. Please make sure to specify a valid timeframe (e.g., 'last month', 'from January 1st to January 31st').\n",
            )
        
        tx_query = TxQuery(
            label=extraction.label,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            direction=extraction.direction,
        )
        txs = self._tx_repository.list_transactions(tx_query)
        
        return TxListResult(answer=format_transactions(txs))