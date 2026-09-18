from dataclasses import dataclass

from langchain.embeddings import Embeddings
from pydantic import BaseModel

from bot.config.prompts_config import PromptConfigs
from bot.config.reranker import RerankerConfig
from bot.config.retriever import RetrieverConfig
from bot.llm.client import LLMClient
from bot.models.memory import SessionState
from bot.models.responses import BotResponse
from bot.models.tx_qa.repository import TransactionsRepository
from bot.routes.doc_qa.embeddings_store import EmbeddingsStore


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