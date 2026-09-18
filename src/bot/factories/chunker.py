from typing import assert_never

from bot.chunker.contextual_chunker import ContextualChunker
from bot.chunker.model import Chunker
from bot.config.chunker import ChunkerConfig, ContextualChunkerConfig


def create_chunker(config: ChunkerConfig) -> Chunker:
    match config:
        case ContextualChunkerConfig():
            return ContextualChunker()
        
        case _:
            assert_never(config)