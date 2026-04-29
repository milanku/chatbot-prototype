from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingsManifest:
    embedding_model: str
    docs_fingerprint: str
    chunk_count: int