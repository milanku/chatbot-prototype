from pydantic import BaseModel


class ReferenceChunkIds(BaseModel):
    required: set[str]
    required_or_relevant: set[str]


class SimulatedPipelineResult(BaseModel):
    retrieved: set[str]
    reranked: set[str]
    judged: set[str]