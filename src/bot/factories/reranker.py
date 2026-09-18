from typing import assert_never

from FlagEmbedding.inference import FlagReranker

from bot.config.reranker import LocalRerankerConfig, RerankerConfig
from bot.routes.doc_qa.reranker import CrossEncoderReranker


def create_reranker(
    config: RerankerConfig,
) -> CrossEncoderReranker:
    match config:
        case LocalRerankerConfig():
            return CrossEncoderReranker(
                reranker=FlagReranker(
                    config.model,
                    use_fp16=config.use_fp16,
                ),
                config=config
            )

        case _:
            assert_never(config)