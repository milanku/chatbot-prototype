from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import QuestionQualityFilterConfig
from rag_eval.questions.models import (
    JudgedQuestion,
    QuestionCollection,
    QuestionQuality,
)


def run_questions_quality_filter(
    *,
    questions: ArtifactRef[QuestionCollection],
    question_judgements: ArtifactRef[list[JudgedQuestion]],
    config: QuestionQualityFilterConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionCollection]:

    def run() -> QuestionCollection:
        filtered_questions_list = [
            question
            for question in questions.data.questions
            if any(
                question_judgement.question_id == question.id
                and question_judgement.quality == QuestionQuality.ACCEPT
                for question_judgement in question_judgements.data
            )
        ]
        filtered_questions = QuestionCollection(
            question_collection_id=questions.data.question_collection_id,
            questions=filtered_questions_list,
        )
        return filtered_questions

    return runner.execute(
        artifact_type=ArtifactType.HQ_QUESTIONS,
        artifact_data_type=QuestionCollection,
        parents=[questions, question_judgements],
        config=config,
        compute=run,
    )
