from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import JudgeConfig
from rag_eval.domain.judgement import QuestionJudgments
from rag_eval.domain.question import QuestionCollection
from rag_eval.factories.llms import get_llm
from rag_eval.judges.question_quality_judge import QuestionQualityJudge
from rag_eval.judges.question_quality_prompt_loader import (
    QuestionQualityJudgePromptLoader,
)


def judge_question_collection(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    config: JudgeConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionJudgments]:
    
    def run() -> QuestionJudgments:
        llm_model = get_llm(config.llm_model)
        question_quality_prompt_loader = QuestionQualityJudgePromptLoader(
            prompt_config=config.prompt_config
        )
        judge = QuestionQualityJudge(
            llm_client=llm_model,
            question_quality_judge_prompt_loader=question_quality_prompt_loader
        )
        question_judgements = judge.judge_questions_quality(
            candidate_questions=question_collection.data.questions
        )
        
        question_judgements = QuestionJudgments(
            question_judgements=question_judgements
        )
        return question_judgements

    return runner.execute(
        artifact_type=ArtifactType.Q_JUDGEMENTS,
        artifact_class=QuestionJudgments,
        parents=[question_collection],
        config=config,
        compute=run,
    )