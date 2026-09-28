from bot.doc_qa.retrieval.judges.chunk_judge import ChunkJudge
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import ChunkJudgePromptLoader
from bot.doc_qa.retrieval.judges.models import (
    ChunkRelevance,
    JudgeOutputFormat,
)
from bot.llm.client import LLMClient


class ChunkRelevanceJudge(ChunkJudge[ChunkRelevance]):
    def __init__(
        self,
        llm_client: LLMClient,
        relevance_judge_prompt_loader: ChunkJudgePromptLoader,
        allow_judgement: frozenset[str],
        output_format: type[JudgeOutputFormat[ChunkRelevance]],
        retries: int = 3,
    ):
        super().__init__(
            llm_client=llm_client,
            relevance_judge_prompt_loader=relevance_judge_prompt_loader,
            judgement_pass_filter=allow_judgement,
            output_format=output_format,
            retries=retries,
        )
