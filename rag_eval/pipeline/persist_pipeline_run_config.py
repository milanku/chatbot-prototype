from rag_eval.artifacts.artifact_lineage import ArtifactNode, ArtifactType
from rag_eval.artifacts.artifact_store import ArtifactStore


def persist_pipeline_run_config(
    *, artifact_id: str, root: ArtifactNode, artifact_store: ArtifactStore
) -> None:
    artifact_store.save(
        artifact_type=ArtifactType.RUNS,
        artifact_id=artifact_id,
        artifact=root,
        artifact_data_type=ArtifactNode,
    )
