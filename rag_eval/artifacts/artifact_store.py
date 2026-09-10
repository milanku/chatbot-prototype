from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from rag_eval.artifacts.ArtifactLineage import ArtifactType

T = TypeVar("T")
ArtifactT = TypeVar("ArtifactT", bound=BaseModel)

class ArtifactStore(Protocol):
    def exists(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str
    ) -> bool:
        ...

    def save(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact: BaseModel | str
    ) -> None:
        ...

    def load(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_class: type[ArtifactT],
    ) -> ArtifactT:
        ...
        
class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_dir_path: Path):
        self._base_path = base_dir_path
        
    def _get_artifact_path(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str
    ) -> Path:
        return self._base_path / artifact_type.value / f"{artifact_id}.json"
    
    def exists(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str
    ) -> bool:
        return self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id
        ).exists()

    def save(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact: BaseModel | str
    ) -> None:
        artifact_path = self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = artifact_path.with_suffix(".tmp")
        
        tmp_path.write_text(
            artifact.model_dump_json(indent=2) if isinstance(artifact, BaseModel) else artifact,
            encoding="utf-8",
        )
        tmp_path.replace(artifact_path)

    def load(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_class: type[ArtifactT],
    ) -> ArtifactT:
        artifact_path = self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")
        return artifact_class.model_validate_json(artifact_path.read_text())