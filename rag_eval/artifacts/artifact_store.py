from typing import Protocol, TypeVar

from rag_eval.artifacts.artifact_lineage import ArtifactType

ArtifactT = TypeVar("ArtifactT")

class ArtifactStore(Protocol):
    def exists(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_data_type: type[ArtifactT],
    ) -> bool:
        ...

    def save(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact: object,
        artifact_data_type: type[ArtifactT],
    ) -> None:
        ...

    def load(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_data_type: type[ArtifactT],
    ) -> ArtifactT:
        ...