from langchain_openai import OpenAIEmbeddings

from bot.models.doc_qa.docs import DocChunk
from bot.models.doc_qa.embedings import EmbeddedDocChunk


def embed_doc_chunks(chunks: list[DocChunk], embedder: OpenAIEmbeddings) -> list[EmbeddedDocChunk]:
    embedded_chunks = []
    embedded_vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    
    for chunk, embedding in zip(chunks, embedded_vectors, strict=True):
        embedded_chunks.append(EmbeddedDocChunk(
            doc_reference=chunk.doc_reference,
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            embedding=embedding
        ))
    return embedded_chunks