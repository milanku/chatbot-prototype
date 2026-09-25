from pathlib import Path

from bot.config.chunker import ContextualChunkerConfig
from bot.config.embedder import LocalEmbedderConfig, SupportedLocalEmbedder
from bot.config.llm import OpenAILLMConfig
from bot.config.models import BotConfig
from bot.config.reranker import LocalRerankerConfig, SupportedLocalReranker
from bot.config.retriever import BM25RetrieverConfig, EmbeddingsRetrieverConfig

BOT_CONFIG = BotConfig(
    docs_dir_path=Path("data/docs"),
    embeddings_dir_path=Path("data/embeddings"),
    transactions_mock_file_path=Path("data/mocks/transactions_mock_jan2025_sep2026.json"),
    llm=OpenAILLMConfig(
        model="gpt-4.1-mini",
    ),
    embedder=LocalEmbedderConfig(model=SupportedLocalEmbedder.JINA_EMBEDDINGS_V3),
    chunker=ContextualChunkerConfig(
        version="contextual_chunker_v01",
    ),
    retrievers=[EmbeddingsRetrieverConfig(top_k=15), BM25RetrieverConfig(top_k=10)],
    reranker=LocalRerankerConfig(
        model=SupportedLocalReranker.BGE_RERANKER_V2_M3, use_fp16=True, top_k=5
    ),
)
