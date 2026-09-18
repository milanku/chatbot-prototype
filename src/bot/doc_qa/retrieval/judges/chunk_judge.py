from enum import StrEnum
from typing import Callable, Generic, TypeVar

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import (
    ChunkJudgePromptInput,
    ChunkJudgePromptLoader,
)
from bot.doc_qa.retrieval.judges.models import (
    ChunkFilter,
    JudgedChunkOutput,
    JudgeOutputFormat,
)
from bot.llm.client import LLMClient

T = TypeVar("T", bound=StrEnum)

class ChunkJudge(ChunkFilter, Generic[T]):
    def __init__(
        self,
        llm_client: LLMClient,
        relevance_judge_prompt_loader: ChunkJudgePromptLoader,
        pass_filter: Callable[[JudgedChunkOutput[T]], bool],
        output_format: type[JudgeOutputFormat[T]],
        retries: int = 3
    ):
        self._llm_client = llm_client
        self._relevance_judge_prompt_loader = relevance_judge_prompt_loader
        self._retries = retries
        self._pass_filter = pass_filter
        self._output_format = output_format
    
    def _check_output_validity(
        self,
        candidate_chunks: list[DocChunk],
        judge_output: JudgeOutputFormat[T]
    ) -> bool:
        expected_ids = {
            chunk.chunk_id
            for chunk in candidate_chunks
        }
        output_ids = {
            output.chunk_id
            for output in judge_output.results
        }
        # There may be duplicates -> Check also the lengths
        return (
            len(candidate_chunks) == len(judge_output.results)
            and expected_ids == output_ids
        )
    
    def _prepare_chunks_for_judge(
        self,
        chunks: list[DocChunk],
    ) -> tuple[list[DocChunk], dict[str, str]]:
        simple_to_original: dict[str, str] = {}
        judge_chunks: list[DocChunk] = []

        for index, chunk in enumerate(chunks, start=1):
            simple_id = f"CHUNK_{index:03}"

            simple_to_original[simple_id] = chunk.chunk_id

            judge_chunks.append(
                chunk.model_copy(
                    update={"chunk_id": simple_id},
                )
            )

        return judge_chunks, simple_to_original
    
    def judge_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[JudgedChunkOutput[T]]:
        # To prevent errors in copying complex ids
        # LLM Judge should see only simple ids (CHUNK_001 etc.)
        # The mapping from simple ids to original ids will be used to translate the judge's output back to the original ids.
        judge_chunks, simple_to_original = self._prepare_chunks_for_judge(
            candidate_chunks
        )
        
        system_prompt = self._relevance_judge_prompt_loader.load_system_instructions()
        prompt_input = ChunkJudgePromptInput(
            candidate_chunks=judge_chunks,
            question=question,
        )
        user_prompt = self._relevance_judge_prompt_loader.build_user_prompt(input=prompt_input)
        
        judge_output = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=self._output_format,
            system_instructions=system_prompt,
            retries=self._retries,
            check_is_output_valid=lambda output: self._check_output_validity(
                candidate_chunks=judge_chunks,
                judge_output=output
            ),
        )
   
        return [
            JudgedChunkOutput(
                chunk_id=simple_to_original[output.chunk_id],
                relevance=output.relevance,
                reason=output.reason,
            )
            for output in judge_output.results
        ]
    
    def filter_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[DocChunk]:
        candidates={
            chunk.chunk_id: chunk for chunk in candidate_chunks
        }
        judged_candidates = self.judge_chunks(
            question=question,
            candidate_chunks=candidate_chunks
        )
        return [candidates[hit.chunk_id] for hit in judged_candidates if self._pass_filter(hit)]