from bot.doc_qa.retrieval.judges.models import ChunkRelevance, JudgedChunk
from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import TestPreparationConfig
from rag_eval.questions.models import QuestionChunkJudgmentsMap, QuestionCollection
from rag_eval.tests.models import RetrievalTestCase


def get_chunks_with_relevance_status(
    chunks: list[JudgedChunk[ChunkRelevance]],
    relevance: ChunkRelevance,
) -> list[JudgedChunk[ChunkRelevance]]:
    return [
        judged_chunk
        for judged_chunk in chunks
        if judged_chunk.relevance == relevance
    ]
    
def run_tests_preparation(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    candidate_retrieval_judgements: ArtifactRef[QuestionChunkJudgmentsMap],
    config: TestPreparationConfig,
    runner: ArtifactStepExecutor
) -> ArtifactRef[list[RetrievalTestCase]]:
    
    def run() -> list[RetrievalTestCase]:
        test_cases: list[RetrievalTestCase] = []
        question_map = {question.id: question for question in question_collection.data.questions}
        for (question_id, judged_chunks) in candidate_retrieval_judgements.data.items():
            test_cases.append(
                RetrievalTestCase(
                    question=question_map[question_id],
                    required_chunks=get_chunks_with_relevance_status(
                        chunks=judged_chunks,
                        relevance=ChunkRelevance.REQUIRED
                    ),
                    relevant_chunks=get_chunks_with_relevance_status(
                        chunks=judged_chunks,
                        relevance=ChunkRelevance.RELEVANT
                    ),
                    irrelevant_chunks=get_chunks_with_relevance_status(
                        chunks=judged_chunks,
                        relevance=ChunkRelevance.IRRELEVANT
                    )
                )
            )
        
        return test_cases
    
    return runner.execute(
        artifact_type=ArtifactType.TESTS,
        artifact_data_type=list[RetrievalTestCase],
        parents=[question_collection, candidate_retrieval_judgements],
        config=config,
        compute=run,
    )