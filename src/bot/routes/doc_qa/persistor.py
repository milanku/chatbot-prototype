import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from bot.models.doc_qa.chunks import EmbeddedDocChunk


class EmbeddingsManifest(BaseModel):
    embedding_model: str
    chunking_version: str
    docs_fingerprint: str
    chunk_count: int

def hash_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def compute_docs_fingerprint(*, file_paths: list[Path]) -> str:
    payload: dict[str, Any] = {
        "files": [
            {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime,
                "hash": hash_file(file_path)
            }
            for file_path in sorted(file_paths)
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

def persist_embeddings(*, output_path: Path, embeddings: list[EmbeddedDocChunk]) -> None:
    # For demonstration save the embeddings as a JSON file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = [
        {
            **chunk.model_dump()
        }
        for chunk in embeddings
    ]

    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
def load_persisted_embeddings(input_path: Path) -> list[EmbeddedDocChunk]:
    try: 
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        doc_chunks = [
            EmbeddedDocChunk.model_validate(item)
            for item in raw
        ]
        return doc_chunks
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at {input_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file at {input_path}:\n {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid transaction data in JSON file at {input_path}:\n {e}")
    
def persist_manifest(*, manifest_path: Path, manifest: EmbeddingsManifest) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_persisted_manifest(manifest_path: Path) -> EmbeddingsManifest | None:
    try:
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings_manifest = EmbeddingsManifest.model_validate(raw_data)
        return embeddings_manifest
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file at {manifest_path}:\n {e}")
    except ValidationError as e:
        raise ValueError(f"Invalid transaction data in JSON file at {manifest_path}:\n {e}")