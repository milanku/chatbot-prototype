from bot.models.doc_qa.chunks import DocChunk, EmbeddedDocChunk
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import EmbeddingsConfig
from rag_eval.domain.docs import DocChunks, EmbeddedDocChunks
from rag_eval.factories.embeddings import get_embeddings


def embed_doc_chunks(
    *,
    doc_chunks: ArtifactRef[DocChunks],
    config: EmbeddingsConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[EmbeddedDocChunks]:
    
    def run() -> EmbeddedDocChunks:
        embedder = get_embeddings(config.embeddings_model)
        
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