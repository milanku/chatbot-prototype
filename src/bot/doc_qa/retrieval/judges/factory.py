from typing import assert_never

from bot.config.judge import (
    ChunkRelevanceJudgeConfig,
    ChunkRequirementJudgeConfig,
    JudgeConfig,
)
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import ChunkJudgePromptLoader
from bot.doc_qa.retrieval.judges.chunk_relevance_judge import ChunkRelevanceJudge
from bot.doc_qa.retrieval.judges.chunk_requirement_judge import ChunkRequirementJudge
from bot.doc_qa.retrieval.judges.models import JudgeOutputFormat
from bot.llm.client import LLMClient


def create_chunk_judge(
    *,
    config: JudgeConfig,
    llm_client: LLMClient,
    relevance_judge_prompt_loader: ChunkJudgePromptLoader,
    allow_judgement: frozenset[str],
):
    match config:
        case ChunkRequirementJudgeConfig():
            return ChunkRequirementJudge(
                llm_client=llm_client,
                relevance_judge_prompt_loader=relevance_judge_prompt_loader,
                allow_judgement=allow_judgement,
                output_format=JudgeOutputFormat,
                retries=config.retries
            )
        case ChunkRelevanceJudgeConfig():
            return ChunkRelevanceJudge(
                llm_client=llm_client,
                relevance_judge_prompt_loader=relevance_judge_prompt_loader,
                allow_judgement=allow_judgement,
                output_format=JudgeOutputFormat,
                retries=config.retries
            )
        case _:
            assert_never(config)