from typing import assert_never

from bot.config.chunker import ChunkerConfig, ContextualChunkerConfig
from bot.doc_qa.indexing.chunkers.contextual_chunker import ContextualChunker
from bot.doc_qa.indexing.models import Chunker


def create_chunker(config: ChunkerConfig) -> Chunker:
    match config:
        case ContextualChunkerConfig():
            return ContextualChunker()

        case _:
            assert_never(config)
