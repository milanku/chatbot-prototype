from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.routes.doc_qa.retriever import (
    BM25ChunksRetriever,
    ChunksRetriever,
    EmbeddingsChunksRetriever,
    RetrievalConfig,
)
from rag_eval.factories.embeddings import get_embeddings


def get_retriever(retriever_model: str, config: RetrievalConfig, embedded_doc_chunks: list[EmbeddedDocChunk]) -> ChunksRetriever:
    e_model = retriever_model
    if e_model == "bm25":
        return BM25ChunksRetriever(
            embedded_doc_chunks=embedded_doc_chunks,
            config=config, 
        )
    elif e_model.startswith("qwen") or e_model.startswith("jina"):
        embedder = get_embeddings(retriever_model)
        return EmbeddingsChunksRetriever(
            embedder=embedder,
            embedded_doc_chunks=embedded_doc_chunks,
            config=config,
        )
    else:
        raise ValueError(f"Unsupported retriever type: {e_model}")