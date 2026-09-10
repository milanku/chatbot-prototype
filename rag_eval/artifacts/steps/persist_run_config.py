from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.artifacts.ArtifactLineage import ArtifactNode, ArtifactType


def persist_run_config(
    *,
    artifact_id: str,
    root: ArtifactNode,
    artifact_store: ArtifactStore
) -> None:
    artifact_store.save(
        artifact_type=ArtifactType.RUNS,
        artifact_id=artifact_id,
        artifact=root
    )