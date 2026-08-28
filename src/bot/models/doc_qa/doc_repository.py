from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.embeddings import Embeddings

from bot.models.doc_qa.chunks import EmbeddedDocChunk


class DocRepository(ABC):   
    @classmethod
    @abstractmethod
    def build_doc_store(
            cls,
            *,
            embedder: Embeddings,
            embedding_model: str,
            embeddings_dir: Path,
            md_docs_dir: Path,
            manifest_path: Path,
            chunking_version: str,) -> "DocRepository":
        pass

    @abstractmethod
    def get_embedded_chunks(self) -> list[EmbeddedDocChunk]:
        """Return the list of embedded document chunks."""
        pass