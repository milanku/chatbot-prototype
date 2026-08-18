from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from bot.logging import log_event
from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.models.doc_qa.embedings import EmbeddingsManifest
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.persistor import (
    build_docs_embeddings,
    compute_docs_fingerprint,
    load_manifest,
    load_persisted_embeddings,
    persist_embeddings,
    persist_manifest,
)


def build_doc_store(
    *,
    embedder: OpenAIEmbeddings,
    md_docs_dir: Path,
    embeddings_dir: Path,
    manifest_path: Path,
    chunking_version: str,
) -> DocStore:
    embedded_chunks = load_or_build_embeddings(
        embedder=embedder,
        chunking_version=chunking_version,
        docs_dir_path=md_docs_dir,
        embeddings_dir_path=embeddings_dir,
        manifest_path=manifest_path,
    )
    return DocStore(embedder=embedder, embedded_chunks=embedded_chunks)

def load_or_build_embeddings(
    *, 
    embedder: OpenAIEmbeddings,
    docs_dir_path: Path,
    embeddings_dir_path: Path,
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
    elif manifest.embedding_model != embedder.model:
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
                "embedding_model": embedder.model,
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
                embedding_model=embedder.model,
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