from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import FilterConfig
from rag_eval.domain.judgement import QuestionJudgments
from rag_eval.domain.question import QuestionCollection


def filter_judged_question_collection(
    *,
    questions: ArtifactRef[QuestionCollection],
    question_judgements: ArtifactRef[QuestionJudgments],
    config: FilterConfig,
    runner: ArtifactStepExecutor
) -> ArtifactRef[QuestionCollection]:
    
    def run() -> QuestionCollection:
        filtered_questions_list = [
            question for question in questions.data.questions
            if any(
                question_judgement.question_id == question.id and question_judgement.quality == config.quality_threshold
                for question_judgement in question_judgements.data.question_judgements
            )
        ]
        filtered_questions = QuestionCollection(
            id=questions.data.id,
            questions=filtered_questions_list
        )
        return filtered_questions

    return runner.execute(
        artifact_type=ArtifactType.HQ_QUESTIONS,
        artifact_class=QuestionCollection,
        parents=[questions, question_judgements],
        config=config,
        compute=run,
    )
    
    