from bot.config.chunker import ChunkerConfig
from bot.doc_qa.indexing.chunkers.factory import create_chunker
from bot.doc_qa.indexing.markdown import MarkdownDocument
from bot.doc_qa.indexing.models import DocChunk
from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor


def run_splitter(
    *,
    md_docs: ArtifactRef[list[MarkdownDocument]],
    config: ChunkerConfig,
    runner: ArtifactStepExecutor,
)-> ArtifactRef[list[DocChunk]]:
    
    def run() -> list[DocChunk]:
        chunker = create_chunker(config)

        return [
            chunk
            for document in md_docs.data
            for chunk in chunker.split(
                document.content,
                str(document.file_path),
            )
        ]

    return runner.execute(
        artifact_type=ArtifactType.DOC_CHUNKS,
        artifact_data_type=list[DocChunk],
        config=config,
        parents=[md_docs],
        compute=run,
    )