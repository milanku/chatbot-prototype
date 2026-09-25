import hashlib
import json

from pydantic import BaseModel
from rag_eval.artifacts.artifact_lineage import ArtifactType


def make_artifact_id(
    *,
    artifact_type: ArtifactType,
    parent_artifact_ids: list[str],
    config: BaseModel | None,
) -> str:
    payload: dict[str, object] = {
        "artifact_type": artifact_type.value,
        "parent_artifact_ids": sorted(parent_artifact_ids),
        "config": config.model_dump(mode="json") if config is not None else None,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
