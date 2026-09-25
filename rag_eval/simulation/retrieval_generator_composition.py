from langchain.embeddings import Embeddings
from rag_eval.config import RetrievalSimulatorConfig
from rag_eval.simulation.retrieval_generator import RetrievalGenerator

from bot.config.reranker import LocalRerankerConfig, SupportedLocalReranker
from bot.config.retriever import (
    BM25RetrieverConfig,
)
from bot.doc_qa.indexing.models import EmbeddedDocChunk
from bot.doc_qa.retrieval.judges.chunk_judge_prompt_loader import ChunkJudgePromptLoader
from bot.doc_qa.retrieval.judges.chunk_requirement_judge import (
    ChunkRequirementJudge,
)
from bot.doc_qa.retrieval.judges.models import (
    ChunkRequirement,
    ChunkRequirementJudgeOutputFormat,
)
from bot.doc_qa.retrieval.rerankers.factory import create_reranker
from bot.doc_qa.retrieval.retrievers.embeddings_retriever import (
    EmbeddingsChunksRetriever,
    EmbeddingsRetrieverConfig,
)
from bot.doc_qa.retrieval.retrievers.factory import (
    create_retriever,
)
from bot.llm.factory import create_llm


def compose_retrieval_generator(
    embedder: Embeddings,
    embedded_doc_chunks: list[EmbeddedDocChunk],
    max_top_k: int,
    config: RetrievalSimulatorConfig,
) -> RetrievalGenerator:
    retriever = EmbeddingsChunksRetriever(
        embedder=embedder,
        embedded_doc_chunks=embedded_doc_chunks,
        config=EmbeddingsRetrieverConfig(top_k=max_top_k),
    )

    reranker = create_reranker(
        config=LocalRerankerConfig(
            model=SupportedLocalReranker.BGE_RERANKER_V2_M3, use_fp16=True, top_k=max_top_k
        )
    )

    bm25_retriever = create_retriever(
        embedder=embedder,
        embedded_doc_chunks=embedded_doc_chunks,
        config=BM25RetrieverConfig(top_k=max_top_k),
    )

    llm_model = create_llm(config.llm_config)

    judge = ChunkRequirementJudge(
        llm_client=llm_model,
        relevance_judge_prompt_loader=ChunkJudgePromptLoader(
            prompt_config=config.relevance_judge_prompt_config
        ),
        allow_judgement=frozenset([ChunkRequirement.REQUIRED, ChunkRequirement.NOT_REQUIRED]),
        output_format=ChunkRequirementJudgeOutputFormat,
        retries=3,
    )

    return RetrievalGenerator(
        embeddings_retriever=retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        two_way_chunk_relevance_judge=judge,
    )
