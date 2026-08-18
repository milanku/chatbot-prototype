from __future__ import annotations

from abc import ABC, abstractmethod

from bot.handlers.models import HandlerResult
from bot.llm.client import LLMClient
from bot.models.memory import SessionState
from bot.models.tx_qa.repository import TransactionsRepository


class Handler(ABC):
    @abstractmethod
    def handle(
        self,
        *,
        message: str,
        session_state: SessionState,
    ) -> HandlerResult:
        raise NotImplementedError


class TxBaseHandler(Handler, ABC):
    def __init__(
        self,
        *,
        tx_repository: TransactionsRepository,
        llm_client: LLMClient,
    ) -> None:
        self._tx_repository = tx_repository
        self._llm_client = llm_client