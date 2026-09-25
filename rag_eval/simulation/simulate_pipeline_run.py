from rag_eval.config import EvalConfig
from rag_eval.evaluator.models import SimulatedPipelineResult
from rag_eval.simulation.retrieval_generator import ChunksRetrieval


def simulate_pipeline_run(
    *,
    precomputed_retrieval: ChunksRetrieval,
    config: EvalConfig,
) -> SimulatedPipelineResult:
    # Simulate configured retrieval cutoffs - use precomputed retrieval results.
    embedding_chunk_ids = {
        chunk.chunk_id for chunk in precomputed_retrieval.retrieved_chunks[: config.retriever_top_k]
    }
    bm25_chunk_ids = {
        chunk.chunk_id for chunk in precomputed_retrieval.bm25_retrieved_chunks[: config.bm25_top_k]
    }
    retrieved_chunk_ids = embedding_chunk_ids | bm25_chunk_ids

    # Reuse the precomputed reranker ordering, but only for chunks
    # that would have been retrieved by this configuration.
    reranked_chunks = [
        chunk
        for chunk in precomputed_retrieval.reranked_chunks
        if chunk.chunk_id in retrieved_chunk_ids
    ][: config.reranker_top_k]

    reranked_chunk_ids = {chunk.chunk_id for chunk in reranked_chunks}

    # Judge decisions were precomputed for all chunks.
    # For this configuration, only reranker survivors reach the judge.
    judged_required_chunk_ids = {
        chunk.chunk.chunk_id for chunk in precomputed_retrieval.chunks_judged_as_required
    }

    judged_chunk_ids = reranked_chunk_ids & judged_required_chunk_ids

    return SimulatedPipelineResult(
        retrieved=retrieved_chunk_ids,
        reranked=reranked_chunk_ids,
        judged=judged_chunk_ids,
    )
