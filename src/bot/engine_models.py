from dataclasses import dataclass

from langchain.embeddings import Embeddings
from pydantic import BaseModel

from bot.bot_models import BotResponse
from bot.config.prompts_config import PromptConfigs
from bot.config.reranker import RerankerConfig
from bot.config.retriever import RetrieverConfig
from bot.doc_qa.indexing.embeddings_store import EmbeddingsStore
from bot.llm.client import LLMClient
from bot.tx_qa.indexing.models import TransactionsRepository
from bot.tx_qa.memory.models import SessionState


@dataclass
class EngineDeps:
    tx_repository: TransactionsRepository
    embeddings_store: EmbeddingsStore
    embedder: Embeddings
    llm_client: LLMClient
    prompt_configs: PromptConfigs
    retriever_configs: list[RetrieverConfig]
    reranker_config: RerankerConfig


class EngineResponse(BaseModel):
    response: BotResponse
    new_state: SessionState | None
