

from enum import StrEnum

from pydantic import BaseModel

from bot.llm.client import LLMClient
from bot.models.doc_qa.chunks import DocChunk
from rag_eval.judges.two_way_chunk_relevance_judge_prompt_loader import (
    TwoWayRelevanceJudgePromptInput,
    TwoWayRelevanceJudgePromptLoader,
)


class TwoWayChunkRelevance(StrEnum):
    REQUIRED = "REQUIRED"
    NOT_REQUIRED = "NOT_REQUIRED"

class JudgedChunkOutput(BaseModel):
    chunk_id: str
    relevance: TwoWayChunkRelevance
    reason: str

class JudgeOutputFormat(BaseModel):
    results: list[JudgedChunkOutput]

class TwoWayChunkRelevanceJudge:
    def __init__(
        self,
        llm_client: LLMClient,
        relevance_judge_prompt_loader: TwoWayRelevanceJudgePromptLoader
    ):
        self._llm_client = llm_client
        self._relevance_judge_prompt_loader = relevance_judge_prompt_loader
        
    async def ajudge_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[JudgedChunkOutput]:
        system_prompt = self._relevance_judge_prompt_loader.load_system_instructions()
        prompt_input = TwoWayRelevanceJudgePromptInput(
            candidate_chunks=candidate_chunks,
            question=question,
        )
        user_prompt = self._relevance_judge_prompt_loader.build_user_prompt(input=prompt_input)
        
        judge_output = JudgeOutputFormat(results=[])
        
        expected_ids = {
            chunk.chunk_id
            for chunk in candidate_chunks
        }
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                judge_output = (
                    await self._llm_client.agenerate_with_structured_output(
                        prompt=user_prompt,
                        output_format=JudgeOutputFormat,
                        system_instructions=system_prompt,
                    )
                )
                judges = {
                    output.chunk_id: output
                    for output in judge_output.results
                }
                
                if set(judges.keys()) == expected_ids:
                    return [
                        JudgedChunkOutput(
                            chunk_id=chunk.chunk_id,
                            relevance=judges[chunk.chunk_id].relevance,
                            reason=judges[chunk.chunk_id].reason,
                        )
                        for chunk in candidate_chunks
                    ]
                
                missing = expected_ids - set(judges.keys())
                unexpected = set(judges.keys()) - expected_ids

                print(
                    f"Attempt {attempt + 1} returned invalid IDs. "
                    f"Missing={missing}, unexpected={unexpected}"
                )
            except Exception as e:
                print(
                    f"Attempt {attempt + 1} failed: {e}"
                )

        raise RuntimeError(
            "Chunk relevance judge failed after 3 attempts"
        )