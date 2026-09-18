from typing import assert_never

from langchain.embeddings import Embeddings

from bot.config.retriever import (
    BM25RetrieverConfig,
    EmbeddingsRetrieverConfig,
    RetrieverConfig,
)
from bot.doc_qa.indexing.models import EmbeddedDocChunk
from bot.doc_qa.retrieval.retrievers.bm25_retriever import BM25ChunksRetriever
from bot.doc_qa.retrieval.retrievers.embeddings_retriever import (
    EmbeddingsChunksRetriever,
)
from bot.doc_qa.retrieval.retrievers.models import ChunksRetriever


def create_retriever(config: RetrieverConfig, embedded_doc_chunks: list[EmbeddedDocChunk], embedder: Embeddings) -> ChunksRetriever:
    match config:
        case BM25RetrieverConfig():
            return BM25ChunksRetriever(config=config, embedded_doc_chunks=embedded_doc_chunks)
        
        case EmbeddingsRetrieverConfig():
            return EmbeddingsChunksRetriever(config=config, embedded_doc_chunks=embedded_doc_chunks, embedder=embedder)
        
        case _:
            assert_never(config)