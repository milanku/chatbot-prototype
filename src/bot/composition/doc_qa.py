from pathlib import Path

from langchain.embeddings import Embeddings

from bot.config.chunker import ChunkerConfig
from bot.config.embedder import EmbedderConfig
from bot.config.prompts_config import PromptConfig
from bot.config.reranker import RerankerConfig
from bot.config.retriever import RetrieverConfig
from bot.doc_qa.coordinator import DocsAnswerCoordinator
from bot.doc_qa.indexing.chunkers.factory import create_chunker
from bot.doc_qa.indexing.embeddings_store_factory import EmbeddingsStoreFactory
from bot.doc_qa.indexing.models import Chunker
from bot.doc_qa.indexing.store_persistor import EmbeddingsStoreRepository
from bot.doc_qa.retrieval.embedders.factory import create_embedder
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import (
    ChunkJudgePromptLoader,
)
from bot.doc_qa.retrieval.judges.chunk_requirement_judge import ChunkRequirementJudge
from bot.doc_qa.retrieval.judges.models import (
    ChunkRequirement,
    ChunkRequirementJudgeOutputFormat,
)
from bot.doc_qa.retrieval.rerankers.factory import create_reranker
from bot.doc_qa.retrieval.retrievers.factory import create_retriever
from bot.doc_qa.synthesis.synthesizer import AnswerSynthesizer
from bot.doc_qa.synthesis.synthesizer_prompt_loader import SynthesizerPromptLoader
from bot.doc_qa.verification.answer_verifier import AnswerVerifier
from bot.doc_qa.verification.claim_extractor import ClaimExtractor
from bot.doc_qa.verification.claim_extractor_prompt_loader import (
    ClaimExtractorPromptLoader,
)
from bot.doc_qa.verification.claim_verifier import ClaimVerifier
from bot.doc_qa.verification.claim_verifier_prompt_loader import ClaimVerifierPromptLoader
from bot.handlers.docs_answer import DocsAnswerHandler
from bot.llm.client import LLMClient


def create_docs_answer_handler(
    *,
    llm_client: LLMClient,
    docs_dir_path: Path,
    embeddings_dir_path: Path,
    embedder_config: EmbedderConfig,
    chunker_config: ChunkerConfig,
    answer_synthesizer_prompt_config: PromptConfig,
    claim_extractor_prompt_config: PromptConfig,
    claim_verifier_prompt_config: PromptConfig,
    chunk_judge_prompt_config: PromptConfig,
    retriever_configs: list[RetrieverConfig],
    reranker_config: RerankerConfig,
) -> DocsAnswerHandler:

    embedder: Embeddings = create_embedder(embedder_config)
    chunker: Chunker = create_chunker(chunker_config)

    embeddings_store = EmbeddingsStoreFactory(
        md_docs_dir=docs_dir_path,
        chunker=chunker,
        chunking_version=chunker_config.version,
        embedder=embedder,
        embeddings_model=embedder_config.model,
        repository=EmbeddingsStoreRepository(
            embeddings_dir_path=embeddings_dir_path,
        ),
    ).load_or_create()

    embedded_doc_chunks = embeddings_store.get_embedded_chunks()

    retrievers = [
        create_retriever(
            config=config,
            embedder=embedder,
            embedded_doc_chunks=embedded_doc_chunks,
        )
        for config in retriever_configs
    ]

    reranker = create_reranker(reranker_config)

    required_chunk_judge = ChunkRequirementJudge(
        llm_client=llm_client,
        relevance_judge_prompt_loader=ChunkJudgePromptLoader(
            prompt_config=chunk_judge_prompt_config,
        ),
        allow_judgement=frozenset([ChunkRequirement.REQUIRED]),
        output_format=ChunkRequirementJudgeOutputFormat,
    )

    synthesizer = AnswerSynthesizer(
        llm_client=llm_client,
        prompt_loader=SynthesizerPromptLoader(prompt_config=answer_synthesizer_prompt_config),
    )

    claim_extractor = ClaimExtractor(
        llm_client=llm_client,
        claim_extraction_prompt_loader=ClaimExtractorPromptLoader(
            prompt_config=claim_extractor_prompt_config
        ),
    )

    claims_verifier = ClaimVerifier(
        llm_client=llm_client,
        claim_verifier_prompt_loader=ClaimVerifierPromptLoader(
            prompt_config=claim_verifier_prompt_config
        ),
    )

    answer_verifier = AnswerVerifier(
        claim_extractor=claim_extractor,
        claim_verifier=claims_verifier,
    )

    coordinator = DocsAnswerCoordinator(
        retrievers=retrievers,
        synthesizer=synthesizer,
        verifier=answer_verifier,
        judge=required_chunk_judge,
        reranker=reranker,
    )

    return DocsAnswerHandler(
        coordinator=coordinator,
    )
