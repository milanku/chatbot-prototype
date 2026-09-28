from rag_eval.evaluator.calc import calc_precision, calc_recall
from rag_eval.evaluator.models import (
    CountMetricAccumulator,
    MetricAccumulator,
    ReferenceChunkIds,
    SimulatedPipelineResult,
    TestCaseEvaluationResult,
)


def _add_stage_metrics(
    *,
    required_chunk_ids: set[str],
    required_or_relevant_chunk_ids: set[str],
    predicted_chunk_ids: set[str],
    required_recall: MetricAccumulator,
    relevant_recall: MetricAccumulator,
    required_precision: MetricAccumulator,
    relevant_precision: MetricAccumulator,
    complete_required_recall: CountMetricAccumulator,
) -> None:
    required_recall.add(calc_recall(required_chunk_ids, predicted_chunk_ids))

    relevant_recall.add(
        calc_recall(
            required_or_relevant_chunk_ids,
            predicted_chunk_ids,
        )
    )

    required_precision.add(calc_precision(required_chunk_ids, predicted_chunk_ids))

    relevant_precision.add(
        calc_precision(
            required_or_relevant_chunk_ids,
            predicted_chunk_ids,
        )
    )

    complete_required_recall.add(required_chunk_ids.issubset(predicted_chunk_ids))


def add_metrics(
    *,
    result: TestCaseEvaluationResult,
    reference: ReferenceChunkIds,
    pipeline: SimulatedPipelineResult,
) -> None:
    _add_stage_metrics(
        required_chunk_ids=reference.required,
        required_or_relevant_chunk_ids=reference.required_or_relevant,
        predicted_chunk_ids=pipeline.retrieved,
        required_recall=result.retrieval_required_recall,
        relevant_recall=result.retrieval_relevant_recall,
        required_precision=result.retrieval_required_precision,
        relevant_precision=result.retrieval_relevant_precision,
        complete_required_recall=result.retrieval_complete_required_recall,
    )

    _add_stage_metrics(
        required_chunk_ids=reference.required,
        required_or_relevant_chunk_ids=reference.required_or_relevant,
        predicted_chunk_ids=pipeline.reranked,
        required_recall=result.reranking_required_recall,
        relevant_recall=result.reranking_relevant_recall,
        required_precision=result.reranking_required_precision,
        relevant_precision=result.reranking_relevant_precision,
        complete_required_recall=result.reranking_complete_required_recall,
    )

    _add_stage_metrics(
        required_chunk_ids=reference.required,
        required_or_relevant_chunk_ids=reference.required_or_relevant,
        predicted_chunk_ids=pipeline.judged,
        required_recall=result.judge_required_recall,
        relevant_recall=result.judge_relevant_recall,
        required_precision=result.judge_required_precision,
        relevant_precision=result.judge_relevant_precision,
        complete_required_recall=result.judge_complete_required_recall,
    )
