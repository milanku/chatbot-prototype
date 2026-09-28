from rag_eval.artifacts.artifact_lineage import ArtifactRef, ArtifactType
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.config import JudgeConfig
from rag_eval.questions.models import (
    QuestionCandidatesDict,
    QuestionChunkJudgmentsMap,
    QuestionCollection,
)

from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import ChunkJudgePromptLoader
from bot.doc_qa.retrieval.judges.chunk_relevance_judge import (
    ChunkRelevance,
    ChunkRelevanceJudge,
)
from bot.doc_qa.retrieval.judges.models import (
    ChunkRelevanceJudgeOutputFormat,
)
from bot.llm.factory import create_llm


def run_candidates_judge(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    candidate_retrieval: ArtifactRef[QuestionCandidatesDict],
    config: JudgeConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[QuestionChunkJudgmentsMap]:

    def run() -> QuestionChunkJudgmentsMap:
        llm_model = create_llm(
            config=config.llm_config,
        )
        relevance_judge_prompt_loader = ChunkJudgePromptLoader(prompt_config=config.prompt_config)
        judge = ChunkRelevanceJudge(
            llm_client=llm_model,
            relevance_judge_prompt_loader=relevance_judge_prompt_loader,
            allow_judgement=frozenset(
                {ChunkRelevance.REQUIRED, ChunkRelevance.RELEVANT, ChunkRelevance.IRRELEVANT}
            ),
            output_format=ChunkRelevanceJudgeOutputFormat,
            retries=3,
        )

        candidate_judgements = {
            question.id: judge.judge_chunks(
                question=question.content, candidate_chunks=candidate_retrieval.data[question.id]
            )
            for question in question_collection.data.questions
        }

        return candidate_judgements

    return runner.execute(
        artifact_type=ArtifactType.JUDGMENTS,
        artifact_data_type=QuestionChunkJudgmentsMap,
        parents=[question_collection, candidate_retrieval],
        config=config,
        compute=run,
    )
