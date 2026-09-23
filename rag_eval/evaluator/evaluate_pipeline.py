from rag_eval.config import EvaluationConfig
from rag_eval.evaluator.metrics import add_metrics
from rag_eval.evaluator.models import (
    ReferenceChunkIds,
    TestCaseEvaluationResult,
)
from rag_eval.simulation.models import QuestionChunksRetrieval
from rag_eval.simulation.simulate_pipeline_run import simulate_pipeline_run
from rag_eval.tests.models import RetrievalTestCase


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

def evaluate_pipeline(
    *,
    test_suite: list[RetrievalTestCase],
    precomputed_retrievals: QuestionChunksRetrieval,
    configs: EvaluationConfig,
) -> list[TestCaseEvaluationResult]:
    evaluation_results = [
        TestCaseEvaluationResult(config=config)
        for config in configs.k_configs
    ]

    for i, test_case in enumerate(test_suite):
        print(
            f"Evaluating test case "
            f"{i + 1}/{len(test_suite)}"
        )

        reference = _build_reference_chunk_ids(test_case)

        for config, result in zip(
            configs.k_configs,
            evaluation_results,
        ):
            pipeline = simulate_pipeline_run(
                precomputed_retrieval=precomputed_retrievals[test_case.question.id],
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