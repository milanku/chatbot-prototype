from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import QuestionGeneratorConfig
from rag_eval.questions.models import QuestionCollection
from rag_eval.questions.question_generator import QuestionGenerator
from rag_eval.tests.testset_generator import compose_testset_generator

from bot.doc_qa.indexing.models import DocChunk


def run_questions_generator(
    *,
    doc_chunks: ArtifactRef[list[DocChunk]],
    config: QuestionGeneratorConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionCollection]:

    def run() -> QuestionCollection:
        testset_generator = compose_testset_generator(config=config, doc_chunks=doc_chunks.data)
        question_generator = QuestionGenerator(config=config, testset_generator=testset_generator)
        question_collection = question_generator.generate_questions()
        return question_collection

    return runner.execute(
        artifact_type=ArtifactType.QUESTIONS,
        artifact_data_type=QuestionCollection,
        parents=[doc_chunks],
        config=config,
        compute=run,
    )
