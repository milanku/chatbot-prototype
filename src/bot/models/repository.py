from dataclasses import dataclass
from datetime import date
from typing import Protocol

from bot.models.tx_qa.domain import Direction, Label, Transaction








    
@dataclass(frozen=True)
class EmbeddedDocChunk(DocChunk):
    embedding: list[float]

@dataclass(frozen=True)
class EmbeddingsManifest:
    embedding_model: str
    docs_fingerprint: str
    chunk_count: int
    
