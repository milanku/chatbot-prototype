from dataclasses import dataclass
from enum import StrEnum

from bot.doc_qa.indexing.models import DocReference


class DocsAnswerStatus(StrEnum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DRAFT_VERIFICATION_FAIL = "DRAFT_VERIFICATION_FAIL"


class InsufficientEvidenceReason(StrEnum):
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    EMPTY_RERANK = "EMPTY_RERANK"
    EMPTY_REQUIRED = "EMPTY_REQUIRED"


@dataclass
class DocsAnswerResult:
    status: DocsAnswerStatus
    answer_text: str
    references: list[DocReference]
