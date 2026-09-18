from dataclasses import dataclass

from bot.models.doc_qa.chunks import DocChunk
from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class ChunkJudgePromptInput:
    question: str
    candidate_chunks: list[DocChunk]
    
class ChunkJudgePromptLoader(PromptLoader[ChunkJudgePromptInput]):
    def build_user_prompt(self, input: ChunkJudgePromptInput) -> str:
        chunks = "\n\n".join(f"Chunk ID: {input_chunk.chunk_id}\nContent: {input_chunk.content}" for input_chunk in input.candidate_chunks)
        
        return (
            "Judge the relevance of the following chunks to the user's question.\n\n"
            f"User Question:\n"
            f"{input.question}\n\n"
            f"Chunks:\n\n"
            f"{chunks}\n"
        )