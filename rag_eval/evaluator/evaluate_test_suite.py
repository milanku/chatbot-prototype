from rag_eval.config import EvalConfig
from rag_eval.domain.evaluation import (
    TestCaseEvaluationResult,
    TestSuiteEvaluationResult,
)
from rag_eval.domain.tests import RetrievalTestCase, RetrievalTestSuite
from rag_eval.evaluator.metrics import add_metrics
from rag_eval.evaluator.models import ReferenceChunkIds
from rag_eval.evaluator.simulate_pipeline_run import simulate_pipeline_run
from rag_eval.test_runner import TestRunnerResults


def _build_reference_chunk_ids(test_case: RetrievalTestCase) -> ReferenceChunkIds:
    required = {
        judged_chunk.chunk.chunk_id
        for judged_chunk in test_case.required_chunks
    }

    relevant = {
        judged_chunk.chunk.chunk_id
        for judged_chunk in test_case.relevant_chunks
    }

    return ReferenceChunkIds(
        required=required,
        required_or_relevant=required | relevant,
    )

def evaluate_test_suite(
    *,
    test_suite: RetrievalTestSuite,
    retrievals: TestRunnerResults,
    configs: list[EvalConfig],
) -> TestSuiteEvaluationResult:
    evaluation_results = TestSuiteEvaluationResult(
        test_case_results=[
            TestCaseEvaluationResult(config=config)
            for config in configs
        ]
    )

    retrievals_by_question_id = {
        retrieval.question_id: retrieval
        for retrieval in retrievals.results
    }

    for i, test_case in enumerate(test_suite.test_cases):
        print(
            f"Evaluating test case "
            f"{i + 1}/{len(test_suite.test_cases)}"
        )

        reference = _build_reference_chunk_ids(test_case)

        retrieval = retrievals_by_question_id[
            test_case.question.id
        ]

        for config, result in zip(
            configs,
            evaluation_results.test_case_results,
        ):
            pipeline = simulate_pipeline_run(
                retrieval=retrieval,
                config=config,
            )

            print(
                f"Config: "
                f"RET={config.retriever_top_k}, "
                f"BM25={config.bm25_top_k}, "
                f"RRK={config.reranker_top_k} | "
                f"chunks: "
                f"retrieved={len(pipeline.retrieved)}, "
                f"reranked={len(pipeline.reranked)}, "
                f"judged={len(pipeline.judged)}"
            )

            add_metrics(
                result=result,
                reference=reference,
                pipeline=pipeline,
            )

    return evaluation_results