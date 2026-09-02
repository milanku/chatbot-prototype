from collections.abc import Callable
from pathlib import Path

from langchain.embeddings import Embeddings

from bot.models.doc_qa.chunks import DocChunk
from bot.routes.doc_qa.read_markdown_docs import read_markdown_docs
from bot.routes.doc_qa.store_models import (
    ChunkEmbedding,
    EmbeddingsStoreData,
    EmbeddingsStoreManifest,
)

Chunker = Callable[[str, str], list[DocChunk]]

class EmbeddingsStore:    
    def __init__(
        self,
        *,
        chunks: dict[str, DocChunk],
        chunk_embeddings: dict[str, ChunkEmbedding],
        manifest: EmbeddingsStoreManifest | None = None,
    ) -> None:
        self._chunks = chunks
        self._chunk_embeddings = chunk_embeddings
        self._manifest = manifest
        
    @classmethod
    def from_docs_dir(
        cls,
        *,
        embedder: Embeddings,
        md_docs_dir: Path,
        manifest: EmbeddingsStoreManifest | None = None,
        chunker: Chunker
    ) -> "EmbeddingsStore":
        md_documents = read_markdown_docs(md_docs_dir)
        all_chunks: list[DocChunk] = [
            chunk
            for md_document in md_documents
            for chunk in chunker(str(md_document.file_path), md_document.content)
        ]
        embedded_vectors = embedder.embed_documents([chunk.content for chunk in all_chunks])
        all_chunk_embeddings: list[ChunkEmbedding] = [
            ChunkEmbedding(chunk_id=chunk.chunk_id, embedding=embedding)
            for chunk, embedding in zip(all_chunks, embedded_vectors, strict=True)
        ]
        return cls(
            chunks={chunk.chunk_id: chunk for chunk in all_chunks},
            chunk_embeddings={ce.chunk_id: ce for ce in all_chunk_embeddings},
            manifest=manifest,
        )
    
    @classmethod
    def from_persistable_model(cls, model: EmbeddingsStoreData) -> "EmbeddingsStore":
        return cls(
            chunks={chunk_id: chunk for chunk_id, chunk in model.chunks.items()},
            chunk_embeddings={ce_id: ce for ce_id, ce in model.chunk_embeddings.items()},
            manifest=model.manifest,
        )
        
    def to_persistable_model(self) -> EmbeddingsStoreData:
        return EmbeddingsStoreData(
            manifest=self._manifest,
            chunks={chunk_id: chunk for chunk_id, chunk in self._chunks.items()},
            chunk_embeddings={ce_id: ce for ce_id, ce in self._chunk_embeddings.items()},
        )
        
    def get_manifest(self) -> EmbeddingsStoreManifest | None:
        return self._manifest
        
    def get_doc_chunks(self) -> list[DocChunk]:
        return list(self._chunks.values())

    def get_chunks_with_embeddings(self) -> list[tuple[DocChunk, ChunkEmbedding]]:
        return [
            (self._chunks[chunk_id], ce)
            for chunk_id, ce in self._chunk_embeddings.items()
        ]

    def get_chunk_embedding(self, chunk_id: str) -> ChunkEmbedding | None:
        return self._chunk_embeddings.get(chunk_id)
    
    def get_chunk_embeddings(self) -> list[ChunkEmbedding]:
        return list(self._chunk_embeddings.values())
    