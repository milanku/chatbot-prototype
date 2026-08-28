from dataclasses import dataclass

from bot.models.memory import SessionState
from bot.models.tx_qa.results import SummaryQueryResult
from bot.routes.tx_qa.explain_summary_parser import TXExplainSummaryParser
from bot.routes.tx_qa.tx_explain_summary_formatter import format_tx_explain_summary


@dataclass
class TxExplainSummaryResult:
    answer: str
    
class TxExplainSummaryCoordinator:
    def __init__(self, *, explain_summary_parser: TXExplainSummaryParser):
        self._explain_summary_parser = explain_summary_parser
        
    def _select_related_summaries(self, *, tx_summaries: tuple[SummaryQueryResult, ...], reference_offset: int, reference_count: int) -> list[SummaryQueryResult]:
        return list(
            tx_summaries[max(0, len(tx_summaries) - reference_offset - reference_count):len(tx_summaries) - reference_offset]
        )
    
    def answer(self, *, question: str, session_state: SessionState) -> TxExplainSummaryResult:
        parse_query_extraction = self._explain_summary_parser.parse(msg=question)
        
        reference_offset = parse_query_extraction.query.reference_offset
        reference_count = parse_query_extraction.query.reference_count

        if reference_offset is None or reference_count is None:
            answer_text = "Sorry, I could not tell which transaction summary you want me to explain. Please ask about the latest result or a specific previous result.\n"
            return TxExplainSummaryResult(
                answer=answer_text
            )
        
        tx_summaries = session_state.tx_summaries
        related_summaries = self._select_related_summaries(
            tx_summaries=tx_summaries,
            reference_offset=reference_offset,
            reference_count=reference_count
        )

        if not related_summaries:
            answer_text = "Sorry, I don't have any transaction summary to explain. Please ask a question about your spending first (e.g., 'How much did I spend on food last month?').\n"
        else:
            answer_text = format_tx_explain_summary(related_summaries)
           
        return TxExplainSummaryResult(
            answer=answer_text
        )