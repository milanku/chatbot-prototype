from bot.config.embedder import EmbedderConfig
from bot.doc_qa.retrieval.embedders.factory import create_embedder
from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.domain.docs import DocChunks, EmbeddedDocChunks


def embed_doc_chunks(
    *,
    doc_chunks: ArtifactRef[DocChunks],
    config: EmbedderConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[EmbeddedDocChunks]:
    
    def run() -> EmbeddedDocChunks:
        embedder = create_embedder(config)
        
        all_chunks: list[DocChunk] = doc_chunks.data.chunks
        embedded_vectors = embedder.embed_documents([chunk.content for chunk in all_chunks])
        embedded_doc_chunks = EmbeddedDocChunks(
            embedded_chunks=[
                EmbeddedDocChunk(**chunk.model_dump(), embedding=embedding)
                for chunk, embedding in zip(all_chunks, embedded_vectors, strict=True)
            ]
        )
        return embedded_doc_chunks

    return runner.execute(
        artifact_type=ArtifactType.EMBEDDINGS,
        artifact_class=EmbeddedDocChunks,
        parents=[doc_chunks],
        config=config,
        compute=run,
    )