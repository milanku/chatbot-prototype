from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import QuestionGeneratorConfig
from rag_eval.domain.docs import DocChunks
from rag_eval.domain.question import QuestionCollection
from rag_eval.factories.testset_generator import get_testset_generator
from rag_eval.generators.question_generator import QuestionGenerator


def generate_question_collection(
    *,
    doc_chunks: ArtifactRef[DocChunks],
    config: QuestionGeneratorConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionCollection]:
    
    def run() -> QuestionCollection:
        testset_generator = get_testset_generator(
            config=config,
            doc_chunks=doc_chunks.data.chunks
        )
        question_generator = QuestionGenerator(
            config=config,
            testset_generator=testset_generator
        )
        question_collection = question_generator.generate_questions()
        return question_collection
    
    return runner.execute(
        artifact_type=ArtifactType.QUESTIONS,
        artifact_class=QuestionCollection,
        parents=[doc_chunks],
        config=config,
        compute=run,
    )