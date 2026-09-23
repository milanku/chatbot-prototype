from langchain.embeddings import Embeddings

from bot.doc_qa.indexing.models import EmbeddedDocChunk
from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import RetrievalSimulatorConfig
from rag_eval.simulation.models import QuestionChunksRetrieval
from rag_eval.simulation.retrieval_generator_composition import (
    compose_retrieval_generator,
)
from rag_eval.tests.models import RetrievalTestCase


def run_retrievals_generation(
    *,
    test_suite: ArtifactRef[list[RetrievalTestCase]],
    embedded_doc_chunks: ArtifactRef[list[EmbeddedDocChunk]],
    embedder: Embeddings,
    max_top_k: int,
    config: RetrievalSimulatorConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionChunksRetrieval]:
    
    def run() -> QuestionChunksRetrieval:
        retrieval_generator = compose_retrieval_generator(
            embedder=embedder,
            embedded_doc_chunks=embedded_doc_chunks.data,
            max_top_k=max_top_k,
            config=config,
        )
        
        precomputed_retrievals = retrieval_generator.precompute_retrieval_sets_for_questions(
            questions=[test_case.question for test_case in test_suite.data],
        )
        return precomputed_retrievals
        
    return runner.execute(
        artifact_type=ArtifactType.RETRIEVAL_STORE,
        artifact_data_type=QuestionChunksRetrieval,
        parents=[test_suite, embedded_doc_chunks],
        config=config,
        compute=run,
    )