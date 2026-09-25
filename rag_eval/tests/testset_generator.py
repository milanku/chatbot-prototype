from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rag_eval.config import QuestionGeneratorConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.transforms import apply_transforms, default_transforms

from bot.doc_qa.indexing.models import DocChunk
from bot.doc_qa.retrieval.embedders.factory import create_embedder


def _create_knowledge_graph(doc_chunks: list[DocChunk]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for chunk in doc_chunks:
        kg.nodes.append(
            Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.doc_reference.file_name,
                    "source_id": chunk.chunk_id,
                },
            )
        )
    return kg


def _create_documents(doc_chunks: list[DocChunk]) -> list[Document]:
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.doc_reference.file_name,
            },
        )
        for chunk in doc_chunks
    ]


def compose_testset_generator(
    config: QuestionGeneratorConfig, *, doc_chunks: list[DocChunk]
) -> TestsetGenerator:
    knowledge_graph = _create_knowledge_graph(doc_chunks)
    docs: list[Document] = _create_documents(doc_chunks)

    embeddings = create_embedder(config.embedder)
    ragas_embedding = LangchainEmbeddingsWrapper(embeddings)
    generator_llm = ChatOpenAI(
        model=config.llm_config.model,
    )
    transformer_llm = LangchainLLMWrapper(generator_llm)
    transforms = default_transforms(
        documents=docs, llm=transformer_llm, embedding_model=ragas_embedding
    )
    apply_transforms(knowledge_graph, transforms)

    return TestsetGenerator(
        llm=transformer_llm,
        embedding_model=ragas_embedding,
        knowledge_graph=knowledge_graph,
        persona_list=[
            Persona(name=person.name, role_description=person.role_description)
            for person in config.persona_list
        ],
    )
