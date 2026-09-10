from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.ArtifactLineage import ArtifactRef, ArtifactType
from rag_eval.config import JudgeConfig
from rag_eval.domain.judgement import CandidateRetrievalJudgments
from rag_eval.domain.question import QuestionCollection
from rag_eval.domain.retrieval import CandidateRetrieval
from rag_eval.factories.llms import get_llm
from rag_eval.judges.chunk_relevance_judge import ChunkRelevanceJudge
from rag_eval.judges.chunk_relevance_prompt_loader import RelevanceJudgePromptLoader


def judge_candidates(
    *,
    question_collection: ArtifactRef[QuestionCollection],
    candidate_retrieval: ArtifactRef[CandidateRetrieval],
    config: JudgeConfig,
    runner: ArtifactStepExecutor,
) -> ArtifactRef[CandidateRetrievalJudgments]:
    
    def run() -> CandidateRetrievalJudgments:
        llm_model = get_llm(config.llm_model)
        relevance_judge_prompt_loader = RelevanceJudgePromptLoader(
            prompt_config=config.prompt_config
        )
        judge = ChunkRelevanceJudge(
            llm_client=llm_model,
            relevance_judge_prompt_loader=relevance_judge_prompt_loader
        )
        candidate_judgements = judge.judge_candidates(
            question_collection=question_collection.data,
            candidate_retrieval=candidate_retrieval.data
        )
        
        candidate_retrieval_judgements = CandidateRetrievalJudgments(
            question_chunk_judgements=candidate_judgements
        )
        return candidate_retrieval_judgements

    return runner.execute(
        artifact_type=ArtifactType.JUDGMENTS,
        artifact_class=CandidateRetrievalJudgments,
        parents=[question_collection, candidate_retrieval],
        config=config,
        compute=run,
    )