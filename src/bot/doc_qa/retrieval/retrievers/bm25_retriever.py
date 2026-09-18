from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from bot.config.retriever import BM25RetrieverConfig
from bot.doc_qa.indexing.models import DocChunk, EmbeddedDocChunk
from bot.doc_qa.retrieval.retrievers.models import ChunksRetriever


class BM25ChunksRetriever(ChunksRetriever):
    def __init__(
        self,
        *,
        embedded_doc_chunks: list[EmbeddedDocChunk],
        config: BM25RetrieverConfig,
    ):
        self._doc_chunks = embedded_doc_chunks
        self._config = config

    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        bm25retriever = BM25Retriever.from_documents(
            documents=[
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id
                    }
                ) for chunk in self._doc_chunks
            ]
        )
        doc_chunks_by_content = {chunk.content: chunk for chunk in self._doc_chunks}
        bm25retriever.k = self._config.top_k
        candidate_chunks_from_bm25 = bm25retriever.invoke(question)
        return [
            DocChunk(
                chunk_id=chunk.metadata['chunk_id'],
                content=chunk.page_content,
                doc_reference=doc_chunks_by_content[chunk.page_content].doc_reference
            )
            for chunk in candidate_chunks_from_bm25
        ][:self._config.top_k]