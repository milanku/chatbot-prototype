from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState
from bot.routes.doc_qa.doc_answer_synthesizer_prompt_loader import (
    DocAnswerSynthesizerPromptLoader,
)
from bot.routes.doc_qa.verifier.claim_extraction_prompt_loader import (
    ClaimExtractionPromptLoader,
)
from bot.routes.doc_qa.verifier.verifier_prompt_loader import (
    ClaimVerifierPromptLoader,
)
from bot.routes.tx_qa.explain_parser_prompt_loader import TXExplainParserPromptLoader
from bot.routes.tx_qa.timeframe_parser_prompt_loader import TimeframeParserPromptLoader
from bot.routing.router_prompt_loader import RouterPromptLoader


@dataclass(frozen=True)
class PromptLoaders:
    router: RouterPromptLoader
    claim_extractor: ClaimExtractionPromptLoader
    claim_verifier: ClaimVerifierPromptLoader
    doc_answer_synthesizer: DocAnswerSynthesizerPromptLoader
    explain_parser: TXExplainParserPromptLoader
    timeframe_parser: TimeframeParserPromptLoader
    
@dataclass(frozen=True)
class RouteHandlerResult:
    answer_text: str
    new_state: SessionState
    references: list[DocReference]
    
class RouteHandler(ABC):
    @abstractmethod
    def handle(
        self,
        *,
        message: str,
        session_state: SessionState,
    ) -> RouteHandlerResult:
        raise NotImplementedError