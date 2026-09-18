from pathlib import Path

from bot.doc_qa.indexing.embeddings_store import EmbeddingsStore
from bot.doc_qa.indexing.store_models import EmbeddingsStoreData


class EmbeddingsStoreRepository:
    def __init__(self, embeddings_dir_path: Path):
        self._dir_path = embeddings_dir_path
    
    def save(self, embeddings_store: EmbeddingsStore) -> None:
        persistable_model = embeddings_store.to_persistable_model()
        (self._dir_path / "persistable_model.json").write_text(persistable_model.model_dump_json())
    
    def load(self) -> EmbeddingsStore | None:
        if not (self._dir_path / "persistable_model.json").exists():
            return None
        persistable_model = EmbeddingsStoreData.model_validate_json(
            (self._dir_path / "persistable_model.json").read_text()
        )
        return EmbeddingsStore.from_persistable_model(persistable_model)