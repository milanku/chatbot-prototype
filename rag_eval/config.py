from pathlib import Path

from pydantic import BaseModel
from ragas.testset.persona import Persona

from bot.config.embedder import EmbedderConfig, LocalEmbedderConfig
from bot.config.prompts_config import PromptConfig
from bot.llm.config import LLMConfig


class MdDocsConfig(BaseModel):
    
    docs_dir: Path
    docs_set_id: Path
    
class ChunkingConfig(BaseModel):
    chunking_version: str

class DocStoreConfig(BaseModel):
    docs_set: MdDocsConfig
    embeddings_dir: str
    embedder: EmbedderConfig
    chunking_version: str

class EmbeddingsConfig(BaseModel):
    embedder: EmbedderConfig
    
class FilterConfig(BaseModel):
    quality_threshold: str = "ACCEPT"
    
class TestPreparationConfig(BaseModel):
    test_suite_id: str
    
class QuestionGeneratorConfig(BaseModel):
    question_set: str
    embedder: EmbedderConfig
    llm_config: LLMConfig
    num_questions: int
    persona_list: list[Persona]
    
class RetrieverForCandidateGeneratorConfig(BaseModel):
    retriever_model: str
    nr_candidates: int
    
class CandidateChunksRetrieverConfig(BaseModel):
    nr_candidates: int
    retriever_models: list[RetrieverForCandidateGeneratorConfig]
    
class JudgeConfig(BaseModel):
    prompt_config: PromptConfig
    llm_config: LLMConfig
    
    
class EvalConfig(BaseModel):
    retriever_top_k: int
    reranker_top_k: int
    bm25_top_k: int
    
class EvaluationConfig(BaseModel):
    version: str
    metrics: list[str]
    k_configs: list[EvalConfig]
    
class EvalRetrievalConfig(BaseModel):
    version: str
    
class RunnerConfig(BaseModel):
    llm_config: LLMConfig
    relevance_judge_prompt_config: PromptConfig

class PipelineConfig(BaseModel):
    md_docs: MdDocsConfig
    chunking_config: ChunkingConfig
    docs_embeddings: EmbedderConfig
    question_generator: QuestionGeneratorConfig
    candidate_answers_generator: CandidateChunksRetrieverConfig
    filter_quality_config: FilterConfig
    test_preparation: TestPreparationConfig
    judge: JudgeConfig
    question_quality_judge: JudgeConfig
    eval_retrieval: EvalRetrievalConfig
    evaluation: EvaluationConfig
    runner: RunnerConfig
    
persona_list = [
    Persona(
        name="Existing customer needing support",
        role_description=(
            "An Aurora Bank customer trying to resolve a banking problem or "
            "understand what to do next. They may ask about failed "
            "transactions, disputes, fraud, account access, complaints, "
            "security, fees, or changes to their account."
        ),
    ),
    Persona(
        name="Prospective customer",
        role_description=(
            "A person considering or applying for Aurora Bank products who "
            "wants to understand eligibility, requirements, costs, terms, "
            "application procedures, and limitations before making a decision."
        ),
    ),
    Persona(
        name="Everyday retail customer",
        role_description=(
            "A retail banking customer who uses Aurora Bank for everyday "
            "banking and asks practical questions about products, accounts, "
            "payments, fees, security, and customer service."
        ),
    ),
    Persona(
        name="Privacy and security conscious customer",
        role_description=(
            "An Aurora Bank customer who is particularly concerned with "
            "privacy, personal data, authentication, online banking security, "
            "fraud prevention, and their responsibilities as a customer."
        ),
    ),
]

def create_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        md_docs=MdDocsConfig(
            docs_dir=Path("rag_eval/data/docs"),
            docs_set_id=Path("docs_001")
        ),
        chunking_config=ChunkingConfig(
            chunking_version="paragraph_with_breadcrumb_chunker_v01"
        ),
        docs_embeddings=LocalEmbedderConfig(
            model="jinaai/jina-embeddings-v3"
        ),
        question_generator=QuestionGeneratorConfig(
            question_set="question_set_001",
            embedder=LocalEmbedderConfig(
                model="qwen/Qwen3-Embedding-4B"
            ),
            llm_config=LLMConfig(
                llm_provider="openai",
                llm_model="gpt-5.4-mini",
            ),
            persona_list=persona_list,
            num_questions=100,
        ),
        candidate_answers_generator=CandidateChunksRetrieverConfig(
            nr_candidates=10,
            retriever_models=[
                RetrieverForCandidateGeneratorConfig(
                    retriever_model="jinaai/jina-embeddings-v3",
                    nr_candidates=10,
                ),
                RetrieverForCandidateGeneratorConfig(
                    retriever_model="qwen/Qwen3-Embedding-4B",
                    nr_candidates=10
                ),
                RetrieverForCandidateGeneratorConfig(
                    retriever_model="bm25",
                    nr_candidates=10,
                )
            ]
        ),
        judge=JudgeConfig(
            prompt_config=PromptConfig(
                directory=Path("rag_eval/prompts"),
                version="three-way-judge-v1",
            ),
            llm_config=LLMConfig(
                llm_provider="openai",
                llm_model="gpt-5.4-mini",
            )
        ),
        question_quality_judge=JudgeConfig(
            prompt_config=PromptConfig(
                directory=Path("rag_eval/prompts"),
                version="question-quality-judge-v1.1",
            ),
            llm_config=LLMConfig(
                llm_provider="openai",
                llm_model="gpt-5.4-mini",
            )
        ),
        eval_retrieval=EvalRetrievalConfig(
            version="v01",
        ),
        runner=RunnerConfig(
            llm_config=LLMConfig(
                llm_provider="openai",
                llm_model="gpt-5-mini",
            ),
            relevance_judge_prompt_config=PromptConfig(
                directory=Path("rag_eval/prompts"),
                version="two-way-judge-v1",
            ),
        ),
        evaluation=EvaluationConfig(
            version="v10-with-bm25",
            metrics=["accuracy"],
            k_configs=[
                EvalConfig(retriever_top_k=rtk, reranker_top_k=rrk, bm25_top_k=bm25k)
                for rtk in [5, 10, 15]
                for rrk in [5, 7, 10, 12]
                for bm25k in [5, 10, 15]
                if rtk >= rrk and rtk >= bm25k
            ]
        ),
        filter_quality_config=FilterConfig(
            quality_threshold="ACCEPT"
        ),
        test_preparation=TestPreparationConfig(
            test_suite_id="test_suite_001"
        ),
    )