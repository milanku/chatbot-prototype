from pydantic import BaseModel, Field

from rag_eval.config import EvalConfig


class MetricAccumulator(BaseModel):
    total: float = 0.0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is not None:
            self.total += value
            self.count += 1

    @property
    def average(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

class TestCaseEvaluationResult(BaseModel):
    config: EvalConfig
    
    retrieval_required_recall: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    retrieval_relevant_recall: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    retrieval_required_precision: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    retrieval_relevant_precision: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )

    reranking_required_recall: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    reranking_relevant_recall: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    reranking_required_precision: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    reranking_relevant_precision: MetricAccumulator = Field(
        default_factory=MetricAccumulator
    )
    
    retrieval_complete_required_recall: int = 0
    reranking_complete_required_recall: int = 0
    
class TestSuiteEvaluationResult(BaseModel):
    test_case_results: list[TestCaseEvaluationResult]