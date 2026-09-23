from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from rag_eval.artifacts.artifact_id_generator import make_artifact_id
from rag_eval.artifacts.artifact_lineage import (
    ArtifactRef,
    ArtifactType,
    TypedArtifactNode,
)
from rag_eval.artifacts.artifact_store import ArtifactStore

DataT = TypeVar("DataT")

class ArtifactStepExecutor:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
    ):
        self._artifact_store = artifact_store
        
    def execute(
        self,
        *,
        artifact_type: ArtifactType,
        artifact_data_type: type[DataT],
        config: BaseModel,
        parents: list[ArtifactRef[Any]],
        compute: Callable[[], DataT],
    ) -> ArtifactRef[DataT]:
        artifact_id = make_artifact_id(
            artifact_type=artifact_type,
            parent_artifact_ids=[parent.artifact_node.artifact_id for parent in parents],
            config=config
        )
        should_run = (
            not self._artifact_store.exists(artifact_type=artifact_type, artifact_id=artifact_id, artifact_data_type=artifact_data_type)
            or any(parent.rerun_downstream for parent in parents)
        )
        if should_run:
            data = compute()
            self._artifact_store.save(
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                artifact=data,
                artifact_data_type=artifact_data_type,
            )
        else:
            data = self._artifact_store.load(
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                artifact_data_type=artifact_data_type
            )
        current_node = TypedArtifactNode(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            config=config
        ) 
        
        for parent in parents:
            parent.artifact_node.add_child(current_node)
        
        return ArtifactRef(
            artifact_node=current_node,
            rerun_downstream=should_run,
            data=data
        )