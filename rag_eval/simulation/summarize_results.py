from typing import Callable

from rag_eval.artifacts.artifact_lineage import ArtifactType
from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.evaluator.format import results_to_dataframes
from rag_eval.evaluator.models import TestCaseEvaluationResult


def summarize_results(
    *,
    artifact_id: str,
    evaluation: list[TestCaseEvaluationResult],
    artifact_store: ArtifactStore
) -> str:
    recall_df, precision_df, complete_recall_df  = results_to_dataframes(evaluation)
    #precision_table = results_precision_to_dataframe(evaluation_results.test_case_results)

    percentage_formatter: Callable[[float], str] = lambda x: f"{x:.1f}%"

    results_str = "\nRECALL"
    results_str += "\n" + recall_df.to_string(
        index=False,
        formatters={
            column: percentage_formatter
            for column in recall_df.columns[3:]
        },
    )
    results_str += "\n" + "-"*40 + "\n"
    results_str += "\nPRECISION"
    results_str += "\n" + precision_df.to_string(
        index=False,
        formatters={
            column: percentage_formatter
            for column in precision_df.columns[3:]
        },
    )
    results_str += "\n" + "-"*40 + "\n"
    results_str += "\nCOMPLETE RECALL"
    results_str += "\n" + complete_recall_df.to_string(
        index=False,
        formatters={
            column: percentage_formatter
            for column in complete_recall_df.columns[3:]
        },
    )
    
    artifact_store.save(
        artifact_type=ArtifactType.SUMMARIES,
        artifact_id=artifact_id,
        artifact=results_str,
        artifact_data_type=str
    )   
    return results_str