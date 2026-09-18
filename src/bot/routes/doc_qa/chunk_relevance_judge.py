from pydantic import BaseModel

from bot.llm.client import LLMClient
from bot.models.doc_qa.chunks import DocChunk
from bot.routes.doc_qa.chunk_relevance_prompt_loader import (
    ChunkRelevanceJudgePromptLoader,
    RelevanceJudgePromptInput,
)
from rag_eval.domain.judgement import (
    ChunkRelevance,
    JudgedChunk,
    QuestionChunkJudgments,
)
from rag_eval.domain.question import QuestionCollection
from rag_eval.domain.retrieval import CandidateRetrieval


class JudgedChunkOutput(BaseModel):
    chunk_id: str
    relevance: ChunkRelevance
    reason: str

class JudgeOutputFormat(BaseModel):
    results: list[JudgedChunkOutput]

class ChunkRelevanceJudge:
    def __init__(
        self,
        llm_client: LLMClient,
        relevance_judge_prompt_loader: ChunkRelevanceJudgePromptLoader
    ):
        self._llm_client = llm_client
        self._relevance_judge_prompt_loader = relevance_judge_prompt_loader
        
    def judge_candidates_for_single_question(
        self,
        question: str,
        candidate_chunks: list[DocChunk],
    ) -> list[JudgedChunk]:
        system_prompt = self._relevance_judge_prompt_loader.load_system_instructions()
        prompt_input = RelevanceJudgePromptInput(
            candidate_chunks=candidate_chunks,
            question=question,
        )
        user_prompt = self._relevance_judge_prompt_loader.build_user_prompt(input=prompt_input)
        
        judge_output = JudgeOutputFormat(results=[])
        
        for _ in range(3):  # Retry up to 3 times
            try:
                judge_output = self._llm_client.generate_with_structured_output(
                    prompt=user_prompt,
                    output_format=JudgeOutputFormat,
                    system_instructions=system_prompt,
                )
                if(len(candidate_chunks) == len(judge_output.results)):
                    break
            except Exception as e:
                print(f"Attempt failed with error: {e}")
        
        judges = {output.chunk_id: output for output in judge_output.results}
        
        
        return [
            JudgedChunk(
                chunk=chunk,
                relevance=judges[chunk.chunk_id].relevance,
                reason=judges[chunk.chunk_id].reason,
            )
            for chunk in candidate_chunks
        ]
        
        
    def judge_candidates(
        self,
        question_collection: QuestionCollection,
        candidate_retrieval: CandidateRetrieval
    ) -> list[QuestionChunkJudgments]:
        judgements: list[QuestionChunkJudgments] = []
        candidate_chunks_by_question_id = {
            candidate_chunks.question_id: candidate_chunks
            for candidate_chunks
            in candidate_retrieval.candidate_chunks
        }
        for question in question_collection.questions:
            candidate_chunks = candidate_chunks_by_question_id[question.id]
            question_judgement = self.judge_candidates_for_single_question(
                question=question.content,
                candidate_chunks=candidate_chunks.chunks,
            )
            judgements.append(
                QuestionChunkJudgments(
                    question_id=question.id,
                    chunk_judgements=question_judgement
                )
            )
        return judgements