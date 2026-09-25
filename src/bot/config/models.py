from pathlib import Path

from pydantic import BaseModel

from bot.config.chunker import ChunkerConfig
from bot.config.embedder import EmbedderConfig
from bot.config.llm import LLMConfig
from bot.config.reranker import RerankerConfig
from bot.config.retriever import RetrieverConfig


class BotConfig(BaseModel):
    docs_dir_path: Path
    embeddings_dir_path: Path
    transactions_mock_file_path: Path

    llm: LLMConfig
    embedder: EmbedderConfig
    chunker: ChunkerConfig
    retrievers: list[RetrieverConfig]
    reranker: RerankerConfig
