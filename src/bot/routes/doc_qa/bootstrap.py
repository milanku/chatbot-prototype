from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from bot.models.doc_qa.embedings import EmbeddedDocChunk
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.persistor import (
    build_doc_embeddings,
    compute_docs_fingerprint,
    load_manifest,
    load_persisted_embeddings,
    persist_embeddings,
)


def build_doc_store(embedder: OpenAIEmbeddings, md_docs_dir: Path, embeddings_dir: Path, manifest_path: Path) -> DocStore:
    embedded_chunks = load_or_build_embeddings(
        embedder=embedder,
        docs_dir_path=md_docs_dir,
        embeddings_dir_path=embeddings_dir,
        manifest_path=manifest_path
    )
    return DocStore(embedder=embedder, embedded_chunks=embedded_chunks)

def load_or_build_embeddings(
    *, 
    embedder: OpenAIEmbeddings,
    docs_dir_path: Path,
    embeddings_dir_path: Path,
    manifest_path: Path
    ) -> list[EmbeddedDocChunk]:
    md_file_paths = list(docs_dir_path.glob("*.md"))
    if not md_file_paths:
        raise ValueError(f"No markdown files found in {docs_dir_path}")
    
    current_embeddings_fingerprint = compute_docs_fingerprint(
        file_paths=md_file_paths,
        embedding_model=embedder.model,
    )
    
    should_rebuild = True
    rebuild_reason = "Missing persisted embeddings or manifest"
    
    if manifest_path.exists() and embeddings_dir_path.exists():
        manifest = load_manifest(manifest_path)
        
        if (manifest.embedding_model == embedder.model and manifest.docs_fingerprint == current_embeddings_fingerprint):
            should_rebuild = False
            rebuild_reason = "Existing persisted embeddings and manifest are up-to-date"
        else:
            rebuild_reason = "Existing manifest or embeddings are outdated"
            
    if should_rebuild:
        embedded_chunks = build_doc_embeddings(
            embedder=embedder,
            docs_dir_path=docs_dir_path,
        )
        persist_embeddings(
            output_path=embeddings_dir_path / "doc_chunks_embeddings.json",
            embeddings=embedded_chunks,
        )
    else:
        embedded_chunks = load_persisted_embeddings(input_path=embeddings_dir_path / "doc_chunks_embeddings.json")
        
    return embedded_chunks