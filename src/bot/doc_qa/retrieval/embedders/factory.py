from typing import assert_never

from langchain.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from bot.config.embedder import (
    EmbedderConfig,
    LocalEmbedderConfig,
    OpenAIEmbedderConfig,
)
from bot.doc_qa.retrieval.embedders.local_embeddings import LocalEmbeddings


def create_embedder(
    config: EmbedderConfig,
) -> Embeddings:
    match config:
        case LocalEmbedderConfig():
            return LocalEmbeddings(
                SentenceTransformer(
                    config.model,
                    trust_remote_code=True,
                )
            )

        case OpenAIEmbedderConfig():
            return OpenAIEmbeddings(
                model=config.model,
            )

        case _:
            assert_never(config)
