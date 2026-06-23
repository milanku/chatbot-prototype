from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings

from bot.models.doc_qa.docs import DocHit, DocRepository
from bot.models.doc_qa.embedings import EmbeddedDocChunk
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass
class DocStore(DocRepository):
    embedder: OpenAIEmbeddings
    embedded_chunks: list[EmbeddedDocChunk]

    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        return [hit.content for hit in self.get_top_k_chunks(query, top_k=top_k)]

    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]:
        query_vector = self.embedder.embed_query(query)

        chunk_scores: list[tuple[EmbeddedDocChunk, float]] = [
            (chunk, vectors_cosine_similarity(query_vector, chunk.embedding))
            for chunk in self.embedded_chunks
        ]

        sorted_chunks = sorted(chunk_scores, key=lambda item: item[1], reverse=True)

        return [
            DocHit(
                id=f"{chunk.doc_reference.file_name}_{chunk.chunk_id}",
                doc_reference=chunk.doc_reference,
                score=score,
                content=chunk.content,
            )
            for chunk, score in sorted_chunks[:top_k]
        ]