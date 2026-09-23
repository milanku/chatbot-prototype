from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import EvaluationConfig
from rag_eval.evaluator.evaluate_pipeline import evaluate_pipeline
from rag_eval.evaluator.models import (
    TestCaseEvaluationResult,
)
from rag_eval.simulation.models import QuestionChunksRetrieval
from rag_eval.tests.models import RetrievalTestCase


def run_pipeline_evaluation_test(
    *,
    test_suite: ArtifactRef[list[RetrievalTestCase]],
    precomputed_retrievals: ArtifactRef[QuestionChunksRetrieval],
    config: EvaluationConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[list[TestCaseEvaluationResult]]:
    
    def run() -> list[TestCaseEvaluationResult]:
        results: list[TestCaseEvaluationResult] = evaluate_pipeline(
            test_suite=test_suite.data,
            configs=config,
            precomputed_retrievals=precomputed_retrievals.data,
        )

        return results
        
    return runner.execute(
        artifact_type=ArtifactType.EVALUATION_RESULTS,
        artifact_data_type=list[TestCaseEvaluationResult],
        parents=[test_suite, precomputed_retrievals],
        config=config,
        compute=run,
    )