from bot.routes.doc_qa.read_markdown_docs import MarkdownDocument
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import ChunkingConfig
from rag_eval.domain.docs import DocChunks
from rag_eval.factories.chunker import get_chunker


def split_md_docs_to_chunks(
    *,
    md_docs: ArtifactRef[list[MarkdownDocument]],
    config: ChunkingConfig,
    runner: ArtifactStepExecutor,
)-> ArtifactRef[DocChunks]:
    
    def run() -> DocChunks:
        chunker = get_chunker(config.chunking_version)

        chunks = [
            chunk
            for document in md_docs.data
            for chunk in chunker(
                str(document.file_path),
                document.content,
            )
        ]

        return DocChunks(
            chunks=chunks,
        )

    return runner.execute(
        artifact_type=ArtifactType.DOC_CHUNKS,
        artifact_class=DocChunks,
        config=config,
        parents=[md_docs],
        compute=run,
    )