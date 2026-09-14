from bot.routes.doc_qa.read_markdown_docs import MarkdownDocument, read_markdown_docs
from bot.routes.doc_qa.utils import calculate_dir_fingerprint
from rag_eval.config import MdDocsConfig
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType, TypedArtifactNode


def read_md_docs(root: ArtifactRef[None], config: MdDocsConfig) -> ArtifactRef[list[MarkdownDocument]]:
    docs_fingerprint = calculate_dir_fingerprint(dir_path=config.docs_dir)
    docs = read_markdown_docs(config.docs_dir / config.docs_set_id)
    
    current_node = TypedArtifactNode(artifact_id=docs_fingerprint, artifact_type=ArtifactType.DOCS, config=config)
    root.artifact_node.add_child(current_node)
    
    return ArtifactRef(
        artifact_node=current_node,
        data=docs,
    )