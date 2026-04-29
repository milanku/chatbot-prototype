from dataclasses import dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from bot.logging import log_event
from bot.models.repository import DocChunk, DocHit, DocRepository, EmbeddedDocChunk
from bot.routes.doc_qa.chunker import split_markdown_into_chunks
from bot.routes.doc_qa.embedder import embed_doc_chunks
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass
class DocStore(DocRepository):
    chunks: list[DocChunk]
    embeddings: list[EmbeddedDocChunk]
    embedder: OpenAIEmbeddings
    
    @classmethod   
    def build_store_from_md_files(cls, embedder: OpenAIEmbeddings, md_file_paths: list[Path]) -> "DocStore":
        all_chunks = []
        embedded_chunks = []
        for file_path in md_file_paths:
            content = file_path.read_text(encoding="utf-8")
            chunks = split_markdown_into_chunks(file_path.name, content)
            all_chunks.extend(chunks)
            embeddings = embed_doc_chunks(chunks=chunks, embedder=embedder)
            embedded_chunks.extend(embeddings)
        log_event(trace_id="1fc4ds1cv51", event="doc_store.load", payload={"num_files": len(md_file_paths), "total_chunks": len(all_chunks), "embedding_length": len(embedded_chunks), "embeddings": [{
            "file_name": chunk.file_name,
            "chunk_id": chunk.chunk_id,
            "embedding": chunk.embedding
        } for chunk in embedded_chunks]})
        return cls(chunks=all_chunks, embedder=embedder, embeddings=embedded_chunks)
    
    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        return [hit.content for hit in self.get_top_k_chunks(query, top_k=top_k)]

    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]:
        query_vector = self.embedder.embed_query(query)
        chunk_scores: list[tuple[DocChunk, float]] = []
        for chunk in self.chunks:
            embedding = next((e.embedding for e in self.embeddings if e.chunk_id == chunk.chunk_id and e.file_name == chunk.file_name), None)
            if embedding is not None:
                score = vectors_cosine_similarity(query_vector, embedding)
                chunk_scores.append((chunk, score))
        sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return [DocHit(id=str(chunk.chunk_id), score=score, content=chunk.content) for chunk, score in sorted_chunks[:top_k]] 