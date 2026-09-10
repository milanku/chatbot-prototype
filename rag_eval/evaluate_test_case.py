from rag_eval.config import EvalConfig
from rag_eval.domain.evaluation import TestCaseEvaluationResult
from rag_eval.domain.tests import RetrievalTestSuite
from rag_eval.test_runner import TestRunner
from rag_eval.utils.calc import calc_precision, calc_recall


def evaluate_test_suite(
    *,
    test_suite: RetrievalTestSuite,
    configs: list[EvalConfig],
    test_runner: TestRunner,
    max_rtk: int,
) -> list[TestCaseEvaluationResult]:
    results = [TestCaseEvaluationResult(config=config) for config in configs]
    
    retrievals = test_runner.run_test_suite(
        questions=[test_case.question for test_case in test_suite.test_cases]
    )

    for i, test_case in enumerate(test_suite.test_cases):
        print(f"Evaluating test case {i + 1}/{len(test_suite.test_cases)}")
        required_ids = {judged_chunk.chunk.chunk_id for judged_chunk in test_case.required_chunks}
        relevant_ids = {judged_chunk.chunk.chunk_id for judged_chunk in test_case.relevant_chunks}
        
        retrieved_docs_all = retrievals[test_case.question.id].retrieved_doc_chunks[:max_rtk]
        bm25_retrieved_docs_all = retrievals[test_case.question.id].bm25_retrieved_doc_chunks[:max_rtk]
        reranked_docs_all = retrievals[test_case.question.id].reranked_doc_chunks[:max_rtk]
     
        # Evaluate each configuration for the current test case
        for config, result in zip(configs, results):
            print(f"Evaluating config: retriever_top_k={config.retriever_top_k}, reranker_top_k={config.reranker_top_k}")
            retrieved_docs = retrieved_docs_all[:config.retriever_top_k]
            retrieved_docs_bm25 = bm25_retrieved_docs_all[:config.bm25_top_k]
            retrieved_docs_ids = {doc.chunk_id for doc in retrieved_docs}
            retrieved_docs_bm25_ids = {doc.chunk_id for doc in retrieved_docs_bm25}
            combined_retrieved_docs_ids = retrieved_docs_ids | retrieved_docs_bm25_ids
            
            
            # Only consider top_k reranked docs that were actually retrieved
            # This aproach simulates the scenario where the reranker can only reorder documents that were actually retrieved by the retriever
            # Reason: Better performance, since it avoids reranking for each configuration separately
            reranked_docs = [
                doc
                for doc in reranked_docs_all
                if doc.chunk_id in combined_retrieved_docs_ids
            ][:config.reranker_top_k]
            reranked_docs_ids = {doc.chunk_id for doc in reranked_docs}
            
            required_or_relevant_ids = required_ids | relevant_ids
            print(f"Retrieved docs IDs: {combined_retrieved_docs_ids}")
            print(f"Reranked docs IDs: {reranked_docs_ids}")
            print(f"Required or relevant IDs: {required_or_relevant_ids}")

            result.retrieval_required_recall.add(
                calc_recall(required_ids, combined_retrieved_docs_ids)
            )
            result.retrieval_relevant_recall.add(
                calc_recall(required_or_relevant_ids, combined_retrieved_docs_ids)
            )
            result.retrieval_required_precision.add(
                calc_precision(required_ids, combined_retrieved_docs_ids)
            )
            result.retrieval_relevant_precision.add(
                calc_precision(required_or_relevant_ids, combined_retrieved_docs_ids)
            )
            result.reranking_required_recall.add(
                calc_recall(required_ids, reranked_docs_ids)
            )
            result.reranking_relevant_recall.add(
                calc_recall(required_or_relevant_ids, reranked_docs_ids)
            )
            result.reranking_required_precision.add(
                calc_precision(required_ids, reranked_docs_ids)
            )
            result.reranking_relevant_precision.add(
                calc_precision(required_or_relevant_ids, reranked_docs_ids)
            )
            result.retrieval_complete_required_recall += required_ids.issubset(combined_retrieved_docs_ids)
            result.reranking_complete_required_recall += required_ids.issubset(reranked_docs_ids)
            
    return results