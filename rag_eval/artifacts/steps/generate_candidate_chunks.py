from bot.routes.doc_qa.retriever import ChunksRetriever, RetrievalConfig
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import CandidateChunksRetrieverConfig
from rag_eval.domain.docs import DocChunks, EmbeddedDocChunks
from rag_eval.domain.question import QuestionCollection
from rag_eval.domain.retrieval import CandidateRetrieval
from rag_eval.factories.retrievers import get_retriever
from rag_eval.generators.candidate_answers_generator import (
    get_candidate_chunks_for_question_collection,
)


def generate_candidate_chunks(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    doc_chunks: ArtifactRef[DocChunks],
    jina_embeddings: ArtifactRef[EmbeddedDocChunks],
    qwen_embeddings: ArtifactRef[EmbeddedDocChunks],
    config: CandidateChunksRetrieverConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[CandidateRetrieval]:
    
    def run() -> CandidateRetrieval:
        retrievers: list[ChunksRetriever] = [
            get_retriever(
                retriever_model="jinaai/jina-embeddings-v3",
                embedded_doc_chunks=jina_embeddings.data.embedded_chunks,
                config=RetrievalConfig(top_k=config.nr_candidates)
            ),
            get_retriever(
                retriever_model="qwen/Qwen3-Embedding-4B",
                embedded_doc_chunks=qwen_embeddings.data.embedded_chunks,
                config=RetrievalConfig(top_k=config.nr_candidates)
            ),
            get_retriever(
                retriever_model="bm25",
                embedded_doc_chunks=jina_embeddings.data.embedded_chunks,
                config=RetrievalConfig(top_k=config.nr_candidates)
            ),
        ]
        
        bank_candidate_chunks = get_candidate_chunks_for_question_collection(
            question_collection=question_collection.data,
            common_doc_chunks={chunk.chunk_id: chunk for chunk in doc_chunks.data.chunks},
            retrievers=retrievers
        )
        
        candidate_chunks = CandidateRetrieval(
            candidate_chunks=bank_candidate_chunks  # Replace with actual candidate chunks generation logic
        )
        return candidate_chunks

    return runner.execute(
        artifact_type=ArtifactType.CANDIDATES,
        artifact_class=CandidateRetrieval,
        parents=[question_collection, doc_chunks, jina_embeddings, qwen_embeddings],
        config=config,
        compute=run,
    )