import re

from rag_eval.config import QuestionGeneratorConfig
from rag_eval.questions.models import Question, QuestionCollection
from ragas.testset import TestsetGenerator


class QuestionGenerator:
    def __init__(self, config: QuestionGeneratorConfig, testset_generator: TestsetGenerator):
        self._config = config
        self._testset_generator = testset_generator

    def _clean_context(self, context: str) -> str:
        return re.sub(r"^<\d+-hop>\n\n", "", context)

    def generate_questions(self) -> QuestionCollection:
        testset_chunks = self._testset_generator.generate(testset_size=self._config.num_questions)

        questions: list[Question] = []
        for sample in testset_chunks.samples:
            user_question = str(sample.eval_sample.user_input)
            reference_answer = str(sample.eval_sample.reference)
            questions.append(
                Question(
                    id=f"question_{len(questions)}",
                    content=user_question,
                    reference_answer=reference_answer,
                )
            )
        return QuestionCollection(
            question_collection_id=self._config.question_set, questions=questions
        )
