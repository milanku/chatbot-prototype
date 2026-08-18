from pathlib import Path

from langchain_core.embeddings import Embeddings

from bot.logging import log_event
from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.models.doc_qa.embedings import EmbeddingsManifest
from bot.routes.doc_qa.persistor import (
    build_docs_embeddings,
    compute_docs_fingerprint,
    load_manifest,
    load_persisted_embeddings,
    persist_embeddings,
    persist_manifest,
)

def load_or_build_embeddings(
    *, 
    embedder: Embeddings,
    embedding_model: str,
    embeddings_dir_path: Path,
    docs_dir_path: Path,
    manifest_path: Path,
    chunking_version: str,
) -> list[EmbeddedDocChunk]:
    md_file_paths = list(docs_dir_path.glob("*.md"))
    if not md_file_paths:
        raise ValueError(f"No markdown files found in {docs_dir_path}")
    
    current_docs_fingerprint = compute_docs_fingerprint(
        file_paths=md_file_paths,
    )
    manifest = load_manifest(manifest_path)
    
    rebuild_reason: str | None = None
    if manifest is None:
        rebuild_reason = "No existing manifest found"
    elif manifest.embedding_model != embedding_model:
        rebuild_reason = "Embedding model has changed"
    elif manifest.chunking_version != chunking_version:
        rebuild_reason = "Chunking version has changed"
    elif manifest.docs_fingerprint != current_docs_fingerprint:
        rebuild_reason = "Docs fingerprint has changed"
        
    if(rebuild_reason is None):
        embedded_chunks = load_persisted_embeddings(input_path=embeddings_dir_path / "doc_chunks_embeddings.json")
    else:
        log_event(
            event="doc_qa.rebuild_doc_embeddings",
            payload={
                "reason": rebuild_reason,
                "embedding_model": embedding_model,
                "chunking_version": chunking_version,
                "docs_fingerprint": current_docs_fingerprint,
            }
        )
        embedded_chunks = build_docs_embeddings(
            embedder=embedder,
            docs_dir_path=docs_dir_path,
        )
        
        persist_manifest(
            manifest_path=manifest_path,
            manifest=EmbeddingsManifest(
                embedding_model=embedding_model,
                chunk_count=len(embedded_chunks),
                chunking_version=chunking_version,
                docs_fingerprint=compute_docs_fingerprint(
                    file_paths=md_file_paths,
                ),
            ),
        )
        persist_embeddings(
            output_path=embeddings_dir_path / "doc_chunks_embeddings.json",
            embeddings=embedded_chunks,
        )
    return embedded_chunks