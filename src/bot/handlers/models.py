from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.config.prompts_config import PromptConfig
from bot.models.doc_qa.references import DocReference
from bot.models.memory import SessionState


@dataclass(frozen=True)
class PromptLoaders:
    router: PromptConfig
    claim_extractor: PromptConfig
    claim_verifier: PromptConfig
    doc_answer_synthesizer: PromptConfig
    explain_parser: PromptConfig
    timeframe_parser: PromptConfig
    
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