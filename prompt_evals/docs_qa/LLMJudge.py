import enum
from dataclasses import dataclass

from pydantic import BaseModel

from bot.llm.client import LLMClient
from bot.models.doc_qa.chunks import DocChunk
from bot.models.prompts import PromptLoader


@dataclass
class EvaluationChunk:
    chunk_id: str
    content: str

class ChunkRelevance(str, enum.Enum):
    REQUIRED="REQUIRED"
    RELEVANT="RELEVANT"
    IRRELEVANT="IRRELEVANT"

class JudgeResult(BaseModel):
    chunk_id: str
    relevance: ChunkRelevance
    reason: str

class JudgeReturn(BaseModel):
    results: list[JudgeResult]

@dataclass(frozen=True)
class RelevanceJudgePromptInput:
    doc_hits: list[DocChunk]
    user_question: str
    reference_answer: str
    
class RelevanceJudgePromptLoader(PromptLoader[RelevanceJudgePromptInput]):
    def build_user_prompt(self, input: RelevanceJudgePromptInput) -> str:
        chunks = "\n\n".join(f"Chunk ID: {input_chunk.chunk_id}\nContent: {input_chunk.content}" for input_chunk in input.doc_hits)
        
        return (
            "Judge the relevance of the following chunks to the user's question.\n\n"
            f"User Question:\n"
            f"{input.user_question}\n\n"
            f"Reference Answer:\n"
            f"{input.reference_answer}\n\n"
            f"Chunks:\n\n"
            f"{chunks}\n"
        )


class LLMJudge:
    def __init__(self, llm_client: LLMClient, relevance_judge_prompt_loader: RelevanceJudgePromptLoader):
        self._llm_client = llm_client
        self._relevance_judge_prompt_loader = relevance_judge_prompt_loader
    
    def judge(self, user_question: str, reference_answer: str, chunk_pool: list[DocChunk]) -> list[JudgeResult]:
        system_prompt = self._relevance_judge_prompt_loader.load_system_instructions()
        prompt_input = RelevanceJudgePromptInput(
            doc_hits=chunk_pool,
            user_question=user_question,
            reference_answer=reference_answer,
        )
        user_prompt = self._relevance_judge_prompt_loader.build_user_prompt(input=prompt_input)
        
        judge_output = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=JudgeReturn,
            system_instructions=system_prompt,
        )
        
        return judge_output.results