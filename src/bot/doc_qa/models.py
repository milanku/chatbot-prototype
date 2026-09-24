from dataclasses import dataclass
from enum import StrEnum

from bot.doc_qa.indexing.models import DocReference


class DocsAnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DRAFT_VERIFICATION_FAIL = "DRAFT_VERIFICATION_FAIL"

@dataclass
class DocsAnswerResult:
    status: DocsAnswerStatus
    answer_text: str
    references: list[DocReference]