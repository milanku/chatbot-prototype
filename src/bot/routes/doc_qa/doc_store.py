from dataclasses import dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from bot.models.repository import DocChunk, DocHit, DocRepository, EmbeddedDocChunk
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass
class DocStore(DocRepository):
    embedder: OpenAIEmbeddings
    embedded_chunks: list[EmbeddedDocChunk]
        
    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        return [hit.content for hit in self.get_top_k_chunks(query, top_k=top_k)]

    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]:
        query_vector = self.embedder.embed_query(query)
        chunk_scores: list[tuple[DocChunk, float]] = []
        for chunk in self.chunks:
            embedding = next((e.embedding for e in self.embedded_chunks if e.chunk_id == chunk.chunk_id and e.file_name == chunk.file_name), None)
            if embedding is not None:
                score = vectors_cosine_similarity(query_vector, embedding)
                chunk_scores.append((chunk, score))
        sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return [DocHit(id=str(chunk.chunk_id), score=score, content=chunk.content) for chunk, score in sorted_chunks[:top_k]]