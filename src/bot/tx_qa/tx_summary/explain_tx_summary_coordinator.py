from dataclasses import dataclass

from bot.tx_qa.memory.models import SessionState
from bot.tx_qa.parsing.explain_summary_parser import ExplainTxSummaryParser
from bot.tx_qa.tx_summary.explain_tx_summary_formatter import format_tx_explain_summary
from bot.tx_qa.tx_summary.models import SummaryQueryResult


@dataclass
class ExplainTxSummaryResult:
    answer: str
    
class ExplainTxSummaryCoordinator:
    def __init__(self, *, explain_summary_parser: ExplainTxSummaryParser):
        self._explain_summary_parser = explain_summary_parser
        
    def _select_related_summaries(self, *, tx_summaries: tuple[SummaryQueryResult, ...], reference_offset: int, reference_count: int) -> list[SummaryQueryResult]:
        return list(
            tx_summaries[max(0, len(tx_summaries) - reference_offset - reference_count):len(tx_summaries) - reference_offset]
        )
    
    def answer(self, *, question: str, session_state: SessionState) -> ExplainTxSummaryResult:
        parse_query_extraction = self._explain_summary_parser.parse(msg=question)
        
        reference_offset = parse_query_extraction.query.reference_offset
        reference_count = parse_query_extraction.query.reference_count

        if reference_offset is None or reference_count is None:
            answer_text = "Sorry, I could not tell which transaction summary you want me to explain. Please ask about the latest result or a specific previous result.\n"
            return ExplainTxSummaryResult(
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
           
        return ExplainTxSummaryResult(
            answer=answer_text
        )