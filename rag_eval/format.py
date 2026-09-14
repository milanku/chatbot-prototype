import pandas as pd

from rag_eval.domain.evaluation import TestCaseEvaluationResult


def results_to_dataframes(
    results: list[TestCaseEvaluationResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config_columns = ["RET(k)", "BM25(k)", "RRK(k)"]

    recall_rows: list[dict[str, float]] = []
    precision_rows: list[dict[str, float]] = []
    complete_recall_rows: list[dict[str, float]] = []

    for result in results:
        config = {
            "RET(k)": result.config.retriever_top_k,
            "BM25(k)": result.config.bm25_top_k,
            "RRK(k)": result.config.reranker_top_k,
        }

        recall_rows.append({
            **config,
           "Required R+B": result.retrieval_required_recall.average or 0.0,
            "Required RRK": result.reranking_required_recall.average or 0.0,
            "Required J": result.judge_required_recall.average or 0.0,
            "Relevant R+B": result.retrieval_relevant_recall.average or 0.0,
            "Relevant RRK": result.reranking_relevant_recall.average or 0.0,
            "Relevant J": result.judge_relevant_recall.average or 0.0,
        })

        precision_rows.append({
            **config,
            "Required R+B": result.retrieval_required_precision.average or 0.0,
            "Required RRK": result.reranking_required_precision.average or 0.0,
            "Required J": result.judge_required_precision.average or 0.0,
            "Relevant R+B": result.retrieval_relevant_precision.average or 0.0,
            "Relevant RRK": result.reranking_relevant_precision.average or 0.0,
            "Relevant J": result.judge_relevant_precision.average or 0.0,
        })

        complete_recall_rows.append({
            **config,
            "R+B": result.retrieval_complete_required_recall.average or 0.0,
            "RRK": result.reranking_complete_required_recall.average or 0.0,
            "J": result.judge_complete_required_recall.average or 0.0,
        })

    recall_df = pd.DataFrame(recall_rows)
    precision_df = pd.DataFrame(precision_rows)
    complete_recall_df = pd.DataFrame(complete_recall_rows)

    # Convert all metric columns to percentages.
    for df in (recall_df, precision_df, complete_recall_df):
        metric_columns = df.columns[len(config_columns):]
        df[metric_columns] = df[metric_columns] * 100

    # Recall / precision share the same column structure.
    metric_columns = pd.MultiIndex.from_tuples([
        ("", "RET(k)"),
        ("", "BM25(k)"),
        ("", "RRK(k)"),
        ("Required", "R+B"),
        ("Required", "RRK"),
        ("Required", "J"),
        ("Relevant", "R+B"),
        ("Relevant", "RRK"),
        ("Relevant", "J"),
    ])

    recall_df.columns = metric_columns
    precision_df.columns = metric_columns

    complete_recall_df.columns = pd.MultiIndex.from_tuples([
        ("", "RET(k)"),
        ("", "BM25(k)"),
        ("", "RRK(k)"),
        ("Complete Required Recall", "R+B"),
        ("Complete Required Recall", "RRK"),
        ("Complete Required Recall", "J"),
    ])

    return recall_df, precision_df, complete_recall_df