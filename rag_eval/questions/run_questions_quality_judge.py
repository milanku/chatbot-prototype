from bot.llm.factory import create_llm
from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import JudgeConfig
from rag_eval.questions.models import JudgedQuestion, QuestionCollection
from rag_eval.questions.question_quality_judge import QuestionQualityJudge
from rag_eval.questions.question_quality_judge_prompt_loader import (
    QuestionQualityJudgePromptLoader,
)


def run_questions_quality_judge(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    config: JudgeConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[list[JudgedQuestion]]:
    
    def run() -> list[JudgedQuestion]:
        llm_model = create_llm(config.llm_config)
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
        
        return question_judgements

    return runner.execute(
        artifact_type=ArtifactType.Q_JUDGEMENTS,
        artifact_data_type=list[JudgedQuestion],
        parents=[question_collection],
        config=config,
        compute=run,
    )