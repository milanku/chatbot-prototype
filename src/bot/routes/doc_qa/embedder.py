from pathlib import Path

from langchain_core.embeddings import Embeddings

from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from bot.routes.doc_qa.chunker import split_markdown_into_chunks


def build_docs_embeddings(
    *,
    embedder: Embeddings,
    docs_dir_path: Path,
) -> list[EmbeddedDocChunk]:
    md_file_paths = list(docs_dir_path.glob("*.md"))
    all_chunks: list[DocChunk] = []
    embedded_chunks: list[EmbeddedDocChunk] = []
    for file_path in md_file_paths:
        content = file_path.read_text(encoding="utf-8")
        chunks = split_markdown_into_chunks(file_path.name, content)
        all_chunks.extend(chunks)
        embeddings = embed_doc_chunks(chunks=chunks, embedder=embedder)
        embedded_chunks.extend(embeddings)
    return embedded_chunks

def embed_doc_chunks(chunks: list[DocChunk], embedder: Embeddings) -> list[EmbeddedDocChunk]:
    embedded_chunks: list[EmbeddedDocChunk] = []
    embedded_vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    
    for chunk, embedding in zip(chunks, embedded_vectors, strict=True):
        embedded_chunks.append(EmbeddedDocChunk(
            doc_reference=chunk.doc_reference,
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            embedding=embedding
        ))
    return embedded_chunks