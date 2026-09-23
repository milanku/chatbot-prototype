from dataclasses import dataclass

from bot.models.prompts import PromptLoader
from rag_eval.questions.models import Question


@dataclass(frozen=True)
class QuestionQualityJudgePromptInput:
    candidate_questions: list[Question]
    
class QuestionQualityJudgePromptLoader(PromptLoader[QuestionQualityJudgePromptInput]):
    def build_user_prompt(self, input: QuestionQualityJudgePromptInput) -> str:
        questions = "\n\n".join(f"Question ID: {input_question.id}\nContent: {input_question.content}" for input_question in input.candidate_questions)
        
        return (
            "Judge the quality of the following questions:\n\n"
            f"{questions}\n"
        )