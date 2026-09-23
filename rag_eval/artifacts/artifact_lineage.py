from __future__ import annotations

import json
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel


class ArtifactType(StrEnum):
    ROOT = "root"
    DOCS = "docs"
    DOC_CHUNKS = "doc_chunks"
    EMBEDDINGS = "embeddings"
    QUESTIONS = "questions"
    CANDIDATES = "candidates"
    TESTS = "tests"
    Q_JUDGEMENTS = "q_judgements"
    HQ_QUESTIONS = "hq_questions"
    RUNS = "runs"
    EVALUATION_RESULTS = "evaluation_results"
    RESULTS = "results"
    SUMMARIES = "summaries"
    JUDGMENTS = "judgments" 
    RETRIEVAL_STORE = "retrieval_store"


ConfigT = TypeVar("ConfigT", bound=BaseModel)

T = TypeVar("T")

class ArtifactRef(BaseModel, Generic[T]):
    artifact_node: ArtifactNode
    rerun_downstream: bool = False
    data: T

class ArtifactNode(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    children: list[ArtifactNode] = []
    
    def add_child(self, child: ArtifactNode) -> None:
        self.children.append(child)
        
class TypedArtifactNode(ArtifactNode, Generic[ConfigT]):
    config: ConfigT
    

def print_artifact_lineage(node: ArtifactNode, indent: int = 0) -> None:
    print(node.model_dump_json(indent=2))
    
# Creates a lineage tree from a JSON string representation of an ArtifactNode.
# Prevents duplicate nodes by using a dictionary to track seen nodes and reusing them when encountered again.
# Example:
#  
#  A — B
#   \   \ 
#    C — D
#
# When serialized to JSON, node D will appear twice, once as a child of B and once as a child of C.
# When deserializing, we want to ensure that both B and C point to the same instance of D, rather than creating two separate instances of D.
def create_lineage_tree_from_json(json_string: str) -> ArtifactNode:
    raw_tree = json.loads(json_string)
    raw_tree = ArtifactNode.model_validate(raw_tree)
    
    seen_nodes = dict[str, ArtifactNode]()
    
    def build_tree(node: ArtifactNode) -> ArtifactNode:
        if node.artifact_id in seen_nodes:
            return seen_nodes[node.artifact_id]

        children_nodes = [build_tree(child) for child in node.children]

        node = ArtifactNode(
            **node.model_dump(exclude={"children"}),
            children=children_nodes
        )
        seen_nodes[node.artifact_id] = node
        return node

    return build_tree(raw_tree)