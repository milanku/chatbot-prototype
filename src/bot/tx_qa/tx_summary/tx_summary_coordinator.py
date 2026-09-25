from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from bot.tx_qa.compute import compute_total_amount
from bot.tx_qa.indexing.models import TransactionsRepository
from bot.tx_qa.memory.models import SessionState
from bot.tx_qa.parsing.models import TxQuery
from bot.tx_qa.parsing.timeframe_parser import TimeframeParser
from bot.tx_qa.timeframe.resolve import resolve_date_range
from bot.tx_qa.tx_summary.models import SummaryQueryResult
from bot.tx_qa.tx_summary.tx_summary_formatter import format_tx_summary


@dataclass
class TxSummaryResult:
    answer_text: str
    new_state: SessionState | None = None


class TxSummaryCoordinator:
    def __init__(self, *, tx_repository: TransactionsRepository, timeframe_parser: TimeframeParser):
        self._tx_repository = tx_repository
        self._timeframe_parser = timeframe_parser

    def answer(self, message: str, *, session_state: SessionState, today: date) -> TxSummaryResult:
        extraction = self._timeframe_parser.parse(message)
        date_range = resolve_date_range(extraction.timeframe, today)

        if date_range is None:
            return TxSummaryResult(
                answer_text="Sorry, I could not determine the date range for your query. Please make sure to specify a valid timeframe (e.g., 'last month', 'from January 1st to January 31st').\n",
            )

        query = TxQuery(
            label=extraction.label,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            direction=extraction.direction,
        )
        txs = self._tx_repository.list_transactions(query)
        total_amount = compute_total_amount(txs)
        query_result: SummaryQueryResult = SummaryQueryResult(
            query=query,
            transactions=txs,
            total=Decimal(total_amount),
            created_at=datetime.now(),
        )
        # Update state with the new query result
        new_state = session_state.add_tx_summary(query_result)
        answer_text = format_tx_summary(query, Decimal(total_amount))
        return TxSummaryResult(
            answer_text=answer_text,
            new_state=new_state,
        )
