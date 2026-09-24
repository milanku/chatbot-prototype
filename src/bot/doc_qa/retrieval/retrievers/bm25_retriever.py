from typing import Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from bot.config.retriever import BM25RetrieverConfig
from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.retrievers.models import ChunksRetriever


class BM25ChunksRetriever(ChunksRetriever):
    def __init__(
        self,
        *,
        doc_chunks: Sequence[DocChunk],
        config: BM25RetrieverConfig,
        retriever: BM25Retriever
    ):
        self._config = config
        self._doc_chunks = doc_chunks
        self._doc_chunks_by_content = {chunk.content: chunk for chunk in self._doc_chunks}
        self._retriever = retriever
        
    @classmethod
    def from_doc_chunks(
        cls,
        *,
        doc_chunks: Sequence[DocChunk],
        config: BM25RetrieverConfig,
    ) -> "BM25ChunksRetriever":
        retriever = BM25Retriever.from_documents(
            documents=[
                Document(
                    page_content=chunk.content,
                    metadata={
                        "chunk_id": chunk.chunk_id
                    }
                ) for chunk in doc_chunks
            ]
        )
       
        return cls(
            doc_chunks=doc_chunks,
            config=config,
            retriever=retriever
        )

    def retrieve(
        self,
        question: str,
    ) -> list[DocChunk]:
        retrieved_chunks = self._retriever.invoke(question)
        
        return [
            DocChunk(
                chunk_id=chunk.metadata["chunk_id"],
                content=chunk.page_content,
                doc_reference=self._doc_chunks_by_content[chunk.page_content].doc_reference
            )
            for chunk in retrieved_chunks
        ][:self._config.top_k]