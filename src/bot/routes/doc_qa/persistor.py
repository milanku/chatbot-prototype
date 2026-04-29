import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from bot.models.repository import DocReference, EmbeddedDocChunk
from bot.routes.doc_qa.chunker import split_markdown_into_chunks
from bot.routes.doc_qa.embedder import embed_doc_chunks
from bot.routes.doc_qa.index import EmbeddingsManifest


def build_doc_embeddings(
    *,
    embedder: OpenAIEmbeddings,
    docs_dir_path: Path,
) -> list[EmbeddedDocChunk]:
    md_file_paths = list(docs_dir_path.glob("*.md"))
    all_chunks = []
    embedded_chunks = []
    for file_path in md_file_paths:
        content = file_path.read_text(encoding="utf-8")
        chunks = split_markdown_into_chunks(file_path.name, content)
        all_chunks.extend(chunks)
        embeddings = embed_doc_chunks(chunks=chunks, embedder=embedder)
        embedded_chunks.extend(embeddings)
    persist_embeddings(output_path=Path("data/embeddings/doc_chunks_embeddings.json"), embeddings=embedded_chunks)
    
    manifest = EmbeddingsManifest(
        embedding_model=embedder.model,
        docs_fingerprint=compute_docs_fingerprint(
            file_paths=md_file_paths,
            embedding_model=embedder.model,
        ),
        chunk_count=len(all_chunks),
    )
    persist_manifest(
        manifest_path=Path("data/embeddings/manifest.json"),
        manifest=manifest,
    )
    return embedded_chunks

def hash_file(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def compute_docs_fingerprint(*, file_paths: list[Path], embedding_model: str) -> str:
    payload = {
        "embedding_model": embedding_model,
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
            **asdict(chunk),
        }
        for chunk in embeddings
    ]

    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
def load_persisted_embeddings(input_path: Path) -> list[EmbeddedDocChunk]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))

    return [
        EmbeddedDocChunk(
            doc_reference=DocReference(**item["doc_reference"]),
            content=item["content"],
            chunk_id=item["chunk_id"],
            embedding=item["embedding"],
        )
        for item in raw
    ]
    
def persist_manifest(*, manifest_path: Path, manifest: EmbeddingsManifest) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
def load_manifest(manifest_path: Path) -> EmbeddingsManifest:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return EmbeddingsManifest(
        embedding_model=raw["embedding_model"],
        docs_fingerprint=raw["docs_fingerprint"],
        chunk_count=raw["chunk_count"],
    )    