from pathlib import Path

from langchain_core.embeddings import Embeddings

from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.routes.doc_qa.bootstrap import load_or_build_embeddings


class DocStore:    
    def __init__(
        self,
        *,
        embedder: Embeddings,
        embedding_model: str,
        embeddings_dir: Path,
        md_docs_dir: Path,
        manifest_path: Path,
        chunking_version: str,
        embedded_chunks: list[EmbeddedDocChunk],
    ) -> None:
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._embeddings_dir = embeddings_dir
        self._md_docs_dir = md_docs_dir
        self._manifest_path = manifest_path
        self._chunking_version = chunking_version
        self._embedded_chunks = embedded_chunks

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

    def get_embedded_chunks(self) -> list[EmbeddedDocChunk]:
        """Return the list of embedded document chunks."""
        return self._embedded_chunks