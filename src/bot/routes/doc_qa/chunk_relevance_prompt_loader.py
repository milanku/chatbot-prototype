from dataclasses import dataclass

from bot.models.doc_qa.chunks import DocChunk
from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class RelevanceJudgePromptInput:
    question: str
    candidate_chunks: list[DocChunk]
    reference_answer: str | None = None
    
class ChunkRelevanceJudgePromptLoader(PromptLoader[RelevanceJudgePromptInput]):
    def build_user_prompt(self, input: RelevanceJudgePromptInput) -> str:
        chunks = "\n\n".join(f"Chunk ID: {input_chunk.chunk_id}\nContent: {input_chunk.content}" for input_chunk in input.candidate_chunks)
        
        ref_answer = f"Reference Answer:\n{input.reference_answer}\n\n" if input.reference_answer else ""
        
        return (
            "Judge the relevance of the following chunks to the user's question.\n\n"
            f"User Question:\n"
            f"{input.question}\n\n"
            f"{ref_answer}"
            f"Chunks:\n\n"
            f"{chunks}\n"
        )