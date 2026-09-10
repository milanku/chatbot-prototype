from FlagEmbedding import FlagReranker
from langchain.embeddings import Embeddings

from bot.config.prompts_config import PromptConfig
from bot.handlers.docs_answer import DocsAnswerHandler
from bot.llm.client import LLMClient
from bot.routes.doc_qa.coordinator import DocsAnswerCoordinator
from bot.routes.doc_qa.doc_answer_synthesizer_prompt_loader import (
    SynthesizerPromptLoader,
)
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.reranker import CrossEncoderReranker
from bot.routes.doc_qa.retriever import ChunksRetriever, RetrievalConfig
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
    doc_store: DocStore,
    answer_synthesizer_prompt_config: PromptConfig,
    claim_extractor_prompt_config: PromptConfig,
    claim_verifier_prompt_config: PromptConfig,
) -> DocsAnswerHandler:

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

    retriever = ChunksRetriever(
        embedder=embedder,
        doc_store=doc_store,
        config=RetrievalConfig(top_k=5),
    )
    
    reranker = CrossEncoderReranker(
        reranker=FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
        )
    )

    coordinator = DocsAnswerCoordinator(
        retriever=retriever,
        synthesizer=synthesizer,
        verifier=answer_verifier,
        reranker=reranker,
    )

    return DocsAnswerHandler(
        coordinator=coordinator,
    )