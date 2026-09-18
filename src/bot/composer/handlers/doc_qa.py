from langchain.embeddings import Embeddings

from bot.config.prompts_config import PromptConfig
from bot.config.reranker import RerankerConfig
from bot.config.retriever import RetrieverConfig
from bot.factories.reranker import create_reranker
from bot.factories.retriever import create_retriever
from bot.handlers.docs_answer import DocsAnswerHandler
from bot.llm.client import LLMClient
from bot.models.doc_qa.chunks import EmbeddedDocChunk
from bot.routes.doc_qa.chunk_relevance_judge import ChunkRelevanceJudge
from bot.routes.doc_qa.chunk_relevance_prompt_loader import (
    ChunkRelevanceJudgePromptLoader,
)
from bot.routes.doc_qa.coordinator import DocsAnswerCoordinator
from bot.routes.doc_qa.doc_answer_synthesizer_prompt_loader import (
    SynthesizerPromptLoader,
)
from bot.routes.doc_qa.synthesizer import AnswerSynthesizer
from bot.routes.doc_qa.verifier.answer_verifier import AnswerVerifier
from bot.routes.doc_qa.verifier.claim_extractor import ClaimExtractor
from bot.routes.doc_qa.verifier.claim_extractor_prompt_loader import (
    ClaimExtractorPromptLoader,
)
from bot.routes.doc_qa.verifier.verifier import ClaimsVerifier
from bot.routes.doc_qa.verifier.verifier_prompt_loader import ClaimVerifierPromptLoader


def create_docs_answer_handler(
    *,
    llm_client: LLMClient,
    embedder: Embeddings,
    embedded_doc_chunks: list[EmbeddedDocChunk],
    answer_synthesizer_prompt_config: PromptConfig,
    claim_extractor_prompt_config: PromptConfig,
    claim_verifier_prompt_config: PromptConfig,
    chunk_relevance_judge_prompt_config: PromptConfig,
    retriever_configs: list[RetrieverConfig],
    reranker_config: RerankerConfig
) -> DocsAnswerHandler:

    retrievers = [create_retriever(
        config=config,
        embedder=embedder,
        embedded_doc_chunks=embedded_doc_chunks,
    ) for config in retriever_configs]
    
    reranker = create_reranker(reranker_config)
    
    chunk_relevance_judge = ChunkRelevanceJudge(
        llm_client=llm_client,
        relevance_judge_prompt_loader=ChunkRelevanceJudgePromptLoader(
            prompt_config=chunk_relevance_judge_prompt_config,
        )
    )
    
    synthesizer = AnswerSynthesizer(
        llm_client=llm_client,
        prompt_loader=SynthesizerPromptLoader(prompt_config=answer_synthesizer_prompt_config),
    )

    claim_extractor = ClaimExtractor(
        llm_client=llm_client,
        claim_extraction_prompt_loader=ClaimExtractorPromptLoader(prompt_config=claim_extractor_prompt_config),
    )

    claims_verifier = ClaimsVerifier(
        llm_client=llm_client,
        claim_verifier_prompt_loader=ClaimVerifierPromptLoader(prompt_config=claim_verifier_prompt_config),
    )

    answer_verifier = AnswerVerifier(
        llm_client=llm_client,
        claim_extractor=claim_extractor,
        claim_verifier=claims_verifier,
    )
    
    coordinator = DocsAnswerCoordinator(
        retrievers=retrievers,
        synthesizer=synthesizer,
        verifier=answer_verifier,
        judge=chunk_relevance_judge,
        reranker=reranker,
    )

    return DocsAnswerHandler(
        coordinator=coordinator,
    )