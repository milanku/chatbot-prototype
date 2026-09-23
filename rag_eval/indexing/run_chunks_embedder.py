from bot.config.embedder import EmbedderConfig
from bot.doc_qa.indexing.models import DocChunk, EmbeddedDocChunk
from bot.doc_qa.retrieval.embedders.factory import create_embedder
from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor


def run_chunks_embedder(
    *,
    chunks: ArtifactRef[list[DocChunk]],
    config: EmbedderConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[list[EmbeddedDocChunk]]:
    
    def run() -> list[EmbeddedDocChunk]:
        embedder = create_embedder(config)
        
        all_chunks: list[DocChunk] = chunks.data
        embedded_vectors = embedder.embed_documents([chunk.content for chunk in all_chunks])
        embedded_doc_chunks = [
            EmbeddedDocChunk(**chunk.model_dump(), embedding=embedding)
            for chunk, embedding in zip(all_chunks, embedded_vectors, strict=True)
        ]
        return embedded_doc_chunks

    return runner.execute(
        artifact_type=ArtifactType.EMBEDDINGS,
        artifact_data_type=list[EmbeddedDocChunk],
        parents=[chunks],
        config=config,
        compute=run,
    )