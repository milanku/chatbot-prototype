from dataclasses import dataclass

from bot.models.doc_qa.chunks import DocChunk
from bot.models.prompts import PromptLoader
from rag_eval.domain.question import Question


@dataclass(frozen=True)
class RelevanceJudgePromptInput:
    question: Question
    candidate_chunks: list[DocChunk]
    
class RelevanceJudgePromptLoader(PromptLoader[RelevanceJudgePromptInput]):
    def build_user_prompt(self, input: RelevanceJudgePromptInput) -> str:
        chunks = "\n\n".join(f"Chunk ID: {input_chunk.chunk_id}\nContent: {input_chunk.content}" for input_chunk in input.candidate_chunks)
        
        return (
            "Judge the relevance of the following chunks to the user's question.\n\n"
            f"User Question:\n"
            f"{input.question.content}\n\n"
            f"Reference Answer:\n"
            f"{input.question.reference_answer}\n\n"
            f"Chunks:\n\n"
            f"{chunks}\n"
        )