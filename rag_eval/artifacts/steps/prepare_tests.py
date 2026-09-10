from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import TestPreparationConfig
from rag_eval.domain.judgement import (
    CandidateRetrievalJudgments,
    ChunkRelevance,
    JudgedChunk,
    QuestionChunkJudgments,
)
from rag_eval.domain.question import QuestionCollection
from rag_eval.domain.tests import RetrievalTestCase, RetrievalTestSuite


def get_chunks_with_relevance_status(
    chunks: list[QuestionChunkJudgments],
    relevance: ChunkRelevance,
) -> list[JudgedChunk]:
    return [
        judged_chunk
        for question_chunk_judgment in chunks
        for judged_chunk in question_chunk_judgment.chunk_judgements
        if judged_chunk.relevance == relevance
    ]
    
def prepare_tests(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    candidate_retrieval_judgements: ArtifactRef[CandidateRetrievalJudgments],
    config: TestPreparationConfig,
    runner: ArtifactStepExecutor
) -> ArtifactRef[RetrievalTestSuite]:
    
    def run() -> RetrievalTestSuite:
        test_cases: list[RetrievalTestCase] = []
        question_map = {question.id: question for question in question_collection.data.questions}
        for question_judgement in candidate_retrieval_judgements.data.question_chunk_judgements:
            test_cases.append(
                RetrievalTestCase(
                    question=question_map[question_judgement.question_id],
                    required_chunks=get_chunks_with_relevance_status(
                        chunks=[question_judgement],
                        relevance=ChunkRelevance.REQUIRED
                    ),
                    relevant_chunks=get_chunks_with_relevance_status(
                        chunks=[question_judgement],
                        relevance=ChunkRelevance.RELEVANT
                    ),
                    irrelevant_chunks=get_chunks_with_relevance_status(
                        chunks=[question_judgement],
                        relevance=ChunkRelevance.IRRELEVANT
                    )
                )
            )
        
        retrieval_test_suite = RetrievalTestSuite(
            test_cases=test_cases
        )
        return retrieval_test_suite
    
    return runner.execute(
        artifact_type=ArtifactType.TESTS,
        artifact_class=RetrievalTestSuite,
        parents=[question_collection, candidate_retrieval_judgements],
        config=config,
        compute=run,
    )