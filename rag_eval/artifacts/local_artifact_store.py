from pathlib import Path

from pydantic import TypeAdapter
from rag_eval.artifacts.artifact_lineage import ArtifactType
from rag_eval.artifacts.artifact_store import ArtifactStore, ArtifactT


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_dir_path: Path):
        self._base_path = base_dir_path

    def _get_artifact_path(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_data_type: type[ArtifactT],
    ) -> Path:
        extension = ".txt" if artifact_data_type is str else ".json"

        return self._base_path / artifact_type.value / f"{artifact_id}{extension}"

    def exists(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_data_type: type[ArtifactT],
    ) -> bool:
        return self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_data_type=artifact_data_type,
        ).exists()

    def save(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact: ArtifactT,
        artifact_data_type: type[ArtifactT],
    ) -> None:
        artifact_path = self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_data_type=artifact_data_type,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = artifact_path.with_suffix(".tmp")

        if artifact_data_type is str and isinstance(artifact, str):
            tmp_path.write_text(
                artifact,
                encoding="utf-8",
            )
        else:
            adapter = TypeAdapter(artifact_data_type)

            tmp_path.write_bytes(
                adapter.dump_json(
                    artifact,
                    indent=2,
                )
            )

        tmp_path.replace(artifact_path)

    def load(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_id: str,
        artifact_data_type: type[ArtifactT],
    ) -> ArtifactT:
        artifact_path = self._get_artifact_path(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            artifact_data_type=artifact_data_type,
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")

        adapter = TypeAdapter(artifact_data_type)
        return adapter.validate_json(artifact_path.read_bytes())
