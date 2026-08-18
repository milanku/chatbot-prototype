from pathlib import Path

from langchain_core.embeddings import Embeddings
from dataclasses import dataclass

from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.models.doc_qa.doc_repository import DocRepository
from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.bootstrap import load_or_build_embeddings
from bot.routes.doc_qa.score import vectors_cosine_similarity

@dataclass
class DocStore(DocRepository):
    embedder: Embeddings
    embedding_model: str
    embeddings_dir: Path
    md_docs_dir: Path
    manifest_path: Path
    chunking_version: str
    embedded_chunks: list[EmbeddedDocChunk]

    @classmethod
    def build_doc_store(
        cls,
        *,
        embedder: Embeddings,
        embedding_model: str,
        embeddings_dir: Path,
        md_docs_dir: Path,
        manifest_path: Path,
        chunking_version: str,
    ) -> "DocStore":
        embedded_chunks = load_or_build_embeddings(
            embedder=embedder,
            embedding_model=embedding_model,
            embeddings_dir_path=embeddings_dir,
            docs_dir_path=md_docs_dir,
            manifest_path=manifest_path,
            chunking_version=chunking_version,
        )
        return DocStore(
            embedder=embedder,
            embedding_model=embedding_model,
            embeddings_dir=embeddings_dir,
            md_docs_dir=md_docs_dir,
            manifest_path=manifest_path,
            chunking_version=chunking_version,
            embedded_chunks=embedded_chunks,
        )

    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]:
        query_vector = self.embedder.embed_query(query)

        chunk_scores: list[tuple[EmbeddedDocChunk, float]] = [
            (chunk, vectors_cosine_similarity(query_vector, chunk.embedding))
            for chunk in self.embedded_chunks
        ]

        sorted_chunks = sorted(chunk_scores, key=lambda item: item[1], reverse=True)

        return [
            DocHit(
                id=f"{chunk.doc_reference.file_name}_{chunk.chunk_id}",
                doc_reference=chunk.doc_reference,
                score=score,
                content=chunk.content,
            )
            for chunk, score in sorted_chunks[:top_k]
        ]