from pydantic import BaseModel
from rag_eval.questions.models import JudgedQuestion, Question, QuestionQuality
from rag_eval.questions.question_quality_judge_prompt_loader import (
    QuestionQualityJudgePromptInput,
    QuestionQualityJudgePromptLoader,
)

from bot.llm.client import LLMClient


class JudgedQuestionllm(BaseModel):
    question_id: str
    quality: QuestionQuality
    reason: str


class JudgeOutputFormat(BaseModel):
    results: list[JudgedQuestionllm]


class QuestionQualityJudge:
    def __init__(
        self,
        llm_client: LLMClient,
        question_quality_judge_prompt_loader: QuestionQualityJudgePromptLoader,
    ):
        self._llm_client = llm_client
        self._question_quality_judge_prompt_loader = question_quality_judge_prompt_loader

    def judge_questions_quality(
        self,
        candidate_questions: list[Question],
    ) -> list[JudgedQuestion]:
        system_prompt = self._question_quality_judge_prompt_loader.load_system_instructions()
        prompt_input = QuestionQualityJudgePromptInput(
            candidate_questions=candidate_questions,
        )
        user_prompt = self._question_quality_judge_prompt_loader.build_user_prompt(
            input=prompt_input
        )

        judge_output = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=JudgeOutputFormat,
            system_instructions=system_prompt,
        )

        judges = {output.question_id: output for output in judge_output.results}

        return [
            JudgedQuestion(
                question_id=question.id,
                quality=judges[question.id].quality,
                reason=judges[question.id].reason,
            )
            for question in candidate_questions
        ]
