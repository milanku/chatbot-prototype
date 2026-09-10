import pandas as pd

from rag_eval.domain.evaluation import TestCaseEvaluationResult


def results_recall_to_dataframe(results: list[TestCaseEvaluationResult]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for result in results:
        rows.append({
            "RET(K)": result.config.retriever_top_k,
            "BM25(K)": result.config.bm25_top_k,
            "RRK(K)": result.config.reranker_top_k,
            "RET.Req.Recall": result.retrieval_required_recall.average or 0.0,
            "RRK.Req.Recall": result.reranking_required_recall.average or 0.0,
            "RET.Rel.Recall": result.retrieval_relevant_recall.average or 0.0,
            "RRK.Rel.Recall": result.reranking_relevant_recall.average or 0.0,
        })
    df = pd.DataFrame(rows)

    # Convert metrics to percentages
    metric_columns = df.columns[3:]
    df[metric_columns] = df[metric_columns] * 100

    return df
        
def results_precision_to_dataframe(results: list[TestCaseEvaluationResult]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for result in results:
        rows.append({
            "RET(K)": result.config.retriever_top_k,
            "BM25(K)": result.config.bm25_top_k,
            "RRK(K)": result.config.reranker_top_k,
            "RET.Req.Precision": result.retrieval_required_precision.average or 0.0,
            "RRK.Req.Precision": result.reranking_required_precision.average or 0.0,
            "RET.Rel.Precision": result.retrieval_relevant_precision.average or 0.0,
            "RRK.Rel.Precision": result.reranking_relevant_precision.average or 0.0,
        })
    df = pd.DataFrame(rows)

    # Convert metrics to percentages
    metric_columns = df.columns[3:]
    df[metric_columns] = df[metric_columns] * 100

    return df