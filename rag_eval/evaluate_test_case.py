from rag_eval.config import EvalConfig
from rag_eval.domain.evaluation import TestCaseEvaluationResult
from rag_eval.domain.tests import RetrievalTestSuite
from rag_eval.test_runner import TestRunnerResults
from rag_eval.utils.calc import calc_precision, calc_recall


def evaluate_test_suite(
    *,
    test_suite: RetrievalTestSuite,
    retrievals: TestRunnerResults,
    configs: list[EvalConfig],
    max_rtk: int,
) -> list[TestCaseEvaluationResult]:
    results = [TestCaseEvaluationResult(config=config) for config in configs]
    
    retrievals_by_question_id = {result.question_id: result for result in retrievals.results}
    
    for i, test_case in enumerate(test_suite.test_cases):
        print(f"Evaluating test case {i + 1}/{len(test_suite.test_cases)}")
        required_chunk_ids = {judged_chunk.chunk.chunk_id for judged_chunk in test_case.required_chunks}
        relevant_chunk_ids = {judged_chunk.chunk.chunk_id for judged_chunk in test_case.relevant_chunks}
        required_or_relevant_chunk_ids = required_chunk_ids | relevant_chunk_ids
        
        embeddings_retrieved_docs = retrievals_by_question_id[test_case.question.id].retrieved_chunks[:max_rtk]
        bm25_retrieved_docs = retrievals_by_question_id[test_case.question.id].bm25_retrieved_chunks[:max_rtk]
        docs_judged_as_required = retrievals_by_question_id[test_case.question.id].chunks_judged_as_required
     
        # Evaluate each configuration for the current test case
        for config, result in zip(configs, results):
            print(f"Evaluating config: retriever_top_k={config.retriever_top_k}, reranker_top_k={config.reranker_top_k}")
            
            test_case_embeddings_chunks = embeddings_retrieved_docs[:config.retriever_top_k]
            test_case_embeddings_chunks_ids = {chunk.chunk_id for chunk in test_case_embeddings_chunks}
            
            test_case_bm25_chunks = bm25_retrieved_docs[:config.bm25_top_k]
            test_case_bm25_chunks_ids = {chunk.chunk_id for chunk in test_case_bm25_chunks}
            
            union_chunks_ids = test_case_embeddings_chunks_ids | test_case_bm25_chunks_ids
            
            # Only consider top_k reranked docs that were actually retrieved
            # This aproach simulates the scenario where the reranker can only reorder documents that were actually retrieved by the retriever
            # Reason: Better performance, since it avoids reranking for each configuration separately
            reranked_chunks = [
                chunk
                for chunk in docs_judged_as_required
                if chunk.chunk_id in union_chunks_ids
            ][:config.reranker_top_k]
            reranked_chunks_ids = {chunk.chunk_id for chunk in reranked_chunks}
            
            print(f"Number of combined retrieved docs: {len(union_chunks_ids)}")
            print(f"Number of reranked docs: {len(reranked_chunks_ids)}")
            print(f"Number of required docs: {len(required_chunk_ids)}")
            print(f"Number of required or relevant docs: {len(required_or_relevant_chunk_ids)}")
            
            result.retrieval_required_recall.add(
                calc_recall(required_chunk_ids, union_chunks_ids)
            )
            result.retrieval_relevant_recall.add(
                calc_recall(required_or_relevant_chunk_ids, union_chunks_ids)
            )
            result.retrieval_required_precision.add(
                calc_precision(required_chunk_ids, union_chunks_ids)
            )
            result.retrieval_relevant_precision.add(
                calc_precision(required_or_relevant_chunk_ids, union_chunks_ids)
            )
            result.reranking_required_recall.add(
                calc_recall(required_chunk_ids, reranked_chunks_ids)
            )
            result.reranking_relevant_recall.add(
                calc_recall(required_or_relevant_chunk_ids, reranked_chunks_ids)
            )
            result.reranking_required_precision.add(
                calc_precision(required_chunk_ids, reranked_chunks_ids)
            )
            result.reranking_relevant_precision.add(
                calc_precision(required_or_relevant_chunk_ids, reranked_chunks_ids)
            )
            result.retrieval_complete_required_recall += required_chunk_ids.issubset(union_chunks_ids)
            result.reranking_complete_required_recall += required_chunk_ids.issubset(reranked_chunks_ids)
            
    return results