from typing import assert_never

from FlagEmbedding.inference import FlagReranker

from bot.config.reranker import LocalRerankerConfig, RerankerConfig
from bot.doc_qa.retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
from bot.doc_qa.retrieval.rerankers.models import Reranker


def create_reranker(
    config: RerankerConfig,
) -> Reranker:
    match config:
        case LocalRerankerConfig():
            return CrossEncoderReranker(
                reranker=FlagReranker(
                    config.model,
                    use_fp16=config.use_fp16,
                ),
                config=config,
            )

        case _:
            assert_never(config)
