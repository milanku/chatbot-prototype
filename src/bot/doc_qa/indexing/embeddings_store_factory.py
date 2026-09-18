from pathlib import Path

from langchain.embeddings import Embeddings

from bot.doc_qa.indexing.embeddings_store import EmbeddingsStore
from bot.doc_qa.indexing.models import Chunker
from bot.doc_qa.indexing.store_models import EmbeddingsStoreManifest
from bot.doc_qa.indexing.store_persistor import (
    EmbeddingsStoreRepository,
)
from bot.doc_qa.indexing.utils import calculate_dir_fingerprint


class EmbeddingsStoreFactory:
    def __init__(
        self,
        *,
        repository: EmbeddingsStoreRepository,
        embedder: Embeddings,
        embeddings_model: str,
        md_docs_dir: Path,
        chunker: Chunker,
        chunking_version: str,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._embeddings_model = embeddings_model
        self._md_docs_dir = md_docs_dir
        self._chunker = chunker
        self._chunking_version = chunking_version

    def load_or_create(self) -> EmbeddingsStore:
        docs_fingerprint = calculate_dir_fingerprint(dir_path=self._md_docs_dir)

        manifest = EmbeddingsStoreManifest(
            embedding_model=self._embeddings_model,
            chunking_version=self._chunking_version,
            docs_fingerprint=docs_fingerprint,
        )

        persisted_store = self._repository.load()

        if persisted_store is not None and persisted_store.get_manifest() == manifest:
            return persisted_store

        store = EmbeddingsStore.from_docs_dir(
            embedder=self._embedder,
            md_docs_dir=self._md_docs_dir,
            chunker=self._chunker,
            manifest=manifest,
        )

        self._repository.save(store)
        return store