from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.domain.evaluation import TestSuiteEvaluationResult
from rag_eval.format import results_precision_to_dataframe, results_recall_to_dataframe


def summarize_results(
    *,
    artifact_id: str,
    evaluation: ArtifactRef[TestSuiteEvaluationResult],
    artifact_store: ArtifactStore
) -> str:
    evaluation_results = evaluation.data
    recall_table = results_recall_to_dataframe(evaluation_results.test_case_results)
    precision_table = results_precision_to_dataframe(evaluation_results.test_case_results)

    results_str = recall_table.to_string(index=False, float_format=lambda x: f"{x:.1f}%")
    results_str += "\n" + "-"*30 + "\n"
    results_str += precision_table.to_string(index=False, float_format=lambda x: f"{x:.1f}%")

    artifact_store.save(
        artifact_type=ArtifactType.SUMMARIES,
        artifact_id=artifact_id,
        artifact=results_str
    )   
    return results_str