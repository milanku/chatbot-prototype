from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import EvaluationConfig
from rag_eval.domain.evaluation import (
    TestCaseEvaluationResult,
    TestSuiteEvaluationResult,
)
from rag_eval.domain.tests import RetrievalTestSuite
from rag_eval.evaluate_test_case import evaluate_test_suite
from rag_eval.test_runner import TestRunnerResults


def evaluate_pipeline(
    *,
    test_suite: ArtifactRef[RetrievalTestSuite],
    retrievals: ArtifactRef[TestRunnerResults],
    config: EvaluationConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[TestSuiteEvaluationResult]:
    
    def run() -> TestSuiteEvaluationResult:
        results: list[TestCaseEvaluationResult] = evaluate_test_suite(
            test_suite=test_suite.data,
            configs=config.k_configs,
            retrievals=retrievals.data,
            max_rtk=max(config.retriever_top_k for config in config.k_configs),
        )
        evaluation_results = TestSuiteEvaluationResult(test_case_results=results)
        
        return evaluation_results
        
    return runner.execute(
        artifact_type=ArtifactType.EVALUATION_RESULTS,
        artifact_class=TestSuiteEvaluationResult,
        parents=[test_suite, retrievals],
        config=config,
        compute=run,
    )