from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import EvalRetrievalConfig
from rag_eval.domain.tests import RetrievalTestSuite
from rag_eval.test_runner import RetrievalRunner, TestRunnerResults


def retrieve(
    *,
    test_suite: ArtifactRef[RetrievalTestSuite],
    retrieval_runner: RetrievalRunner,
    config: EvalRetrievalConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[TestRunnerResults]:
    
    def run() -> TestRunnerResults:
        retrievals = retrieval_runner.run_test_suite(
            questions=[test_case.question for test_case in test_suite.data.test_cases]
        )
        return retrievals
        
    return runner.execute(
        artifact_type=ArtifactType.EVAL_RETRIEVALS,
        artifact_class=TestRunnerResults,
        parents=[test_suite],
        config=config,
        compute=run,
    )