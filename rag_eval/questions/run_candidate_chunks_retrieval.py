from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import CandidateChunksRetrieverConfig
from rag_eval.questions.candidate_answers_generator import (
    retrieve_candidate_chunks_for_question_collection,
)
from rag_eval.questions.models import QuestionCandidatesDict, QuestionCollection

from bot.doc_qa.indexing.models import DocChunk, EmbeddedDocChunk
from bot.doc_qa.retrieval.embedders.factory import create_embedder
from bot.doc_qa.retrieval.retrievers.factory import (
    ChunksRetriever,
    create_retriever,
)


def run_candidate_chunks_retrieval(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    doc_chunks: ArtifactRef[list[DocChunk]],
    jina_embeddings: ArtifactRef[list[EmbeddedDocChunk]],
    qwen_embeddings: ArtifactRef[list[EmbeddedDocChunk]],
    config: CandidateChunksRetrieverConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionCandidatesDict]:

    def run() -> QuestionCandidatesDict:
        retrievers: list[ChunksRetriever] = [
            create_retriever(
                config=retriever_config.config,
                embedded_doc_chunks=chunks,
                embedder=create_embedder(retriever_config.embedder_config),
            )
            for retriever_config, chunks in zip(
                config.retriever_configs,
                [jina_embeddings.data, qwen_embeddings.data, jina_embeddings.data],
                strict=True,
            )
        ]

        candidate_chunks_bank = retrieve_candidate_chunks_for_question_collection(
            question_collection=question_collection.data, retrievers=retrievers
        )
        return candidate_chunks_bank

    return runner.execute(
        artifact_type=ArtifactType.CANDIDATES,
        artifact_data_type=QuestionCandidatesDict,
        parents=[question_collection, doc_chunks, jina_embeddings, qwen_embeddings],
        config=config,
        compute=run,
    )
