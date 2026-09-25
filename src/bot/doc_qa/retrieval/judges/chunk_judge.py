from abc import ABC
from enum import StrEnum
from typing import Generic, TypeVar

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import (
    ChunkJudgePromptInput,
    ChunkJudgePromptLoader,
)
from bot.doc_qa.retrieval.judges.models import (
    ChunkFilter,
    JudgedChunk,
    JudgeOutputFormat,
)
from bot.llm.client import LLMClient
from bot.logging import log_event

T = TypeVar("T", bound=StrEnum)


class ChunkJudge(ChunkFilter, Generic[T], ABC):
    def __init__(
        self,
        llm_client: LLMClient,
        relevance_judge_prompt_loader: ChunkJudgePromptLoader,
        judgement_pass_filter: frozenset[str],
        output_format: type[JudgeOutputFormat[T]],
        retries: int = 3,
    ):
        self._llm_client = llm_client
        self._relevance_judge_prompt_loader = relevance_judge_prompt_loader
        self._retries = retries
        self._judgement_pass_filter = judgement_pass_filter
        self._output_format = output_format

    def _check_output_validity(
        self, candidate_chunks: list[DocChunk], judge_output: JudgeOutputFormat[T]
    ) -> bool:
        """Check if all chunks were processed by the judge

        Args:
            candidate_chunks (list[DocChunk]): list of chunks
            judge_output (JudgeOutputFormat[T]): output from the judge

        Returns:
            bool: True if all chunks were processed by the judge, False otherwise.
        """

        expected_ids = {chunk.chunk_id for chunk in candidate_chunks}
        judge_output_ids = {output.chunk_id for output in judge_output.results}
        # There may be duplicates -> Check also the lengths
        if len(candidate_chunks) != len(judge_output.results) or expected_ids != judge_output_ids:
            log_event(
                event="judge_output_invalid",
                payload={
                    "expected_ids": expected_ids,
                    "output_ids": judge_output_ids,
                    "candidate_chunks_count": len(candidate_chunks),
                    "judge_output_count": len(judge_output.results),
                },
            )
            return False

        return True

    def _prepare_chunks_for_judge(
        self,
        chunks: list[DocChunk],
    ) -> tuple[list[DocChunk], dict[str, DocChunk]]:
        simple_ids_to_original_chunks: dict[str, DocChunk] = {}
        chunks_with_simple_ids: list[DocChunk] = []

        for index, chunk in enumerate(chunks, start=1):
            simple_id = f"CHUNK_{index:03}"

            simple_ids_to_original_chunks[simple_id] = chunk

            chunks_with_simple_ids.append(
                chunk.model_copy(
                    update={"chunk_id": simple_id},
                )
            )

        return chunks_with_simple_ids, simple_ids_to_original_chunks

    def judge_chunks(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[JudgedChunk[T]]:
        # To prevent errors in copying complex ids by LLM
        # LLM Judge should see only simple ids (CHUNK_001 etc.)
        # The mapping from simple ids to original chunks will be used to restore the judge's output back to the original ids.
        chunks_with_simple_ids, simple_ids_to_original_chunks = self._prepare_chunks_for_judge(
            candidate_chunks
        )

        system_prompt = self._relevance_judge_prompt_loader.load_system_instructions()
        prompt_input = ChunkJudgePromptInput(
            candidate_chunks=chunks_with_simple_ids,
            question=question,
        )
        user_prompt = self._relevance_judge_prompt_loader.build_user_prompt(input=prompt_input)

        judge_output = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            system_instructions=system_prompt,
            output_format=self._output_format,
            retries=self._retries,
            check_is_output_valid=lambda output: self._check_output_validity(
                candidate_chunks=chunks_with_simple_ids, judge_output=output
            ),
        )

        return [
            JudgedChunk(
                chunk=simple_ids_to_original_chunks[output.chunk_id],
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
        # Filter is set via judgement_pass_filter
        candidates = {chunk.chunk_id: chunk for chunk in candidate_chunks}
        judged_candidates = self.judge_chunks(question=question, candidate_chunks=candidate_chunks)
        return [
            candidates[hit.chunk.chunk_id]
            for hit in judged_candidates
            if hit.relevance in self._judgement_pass_filter
        ]
