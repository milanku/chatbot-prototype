from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.transforms import apply_transforms, default_transforms

from bot.factories.embedder import create_embedder
from bot.models.doc_qa.chunks import DocChunk
from rag_eval.config import QuestionGeneratorConfig


def get_testset_generator(config: QuestionGeneratorConfig, *, doc_chunks: list[DocChunk]) -> TestsetGenerator:
    kg = KnowledgeGraph()
    docs: list[Document] = [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.doc_reference.file_name,
            }
        ) for chunk in doc_chunks
    ]
    for chunk in doc_chunks:
        kg.nodes.append(
            Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.doc_reference.file_name,
                    "source_id": chunk.chunk_id
                }
            )
        )
    generator_llm = ChatOpenAI(
        model=config.llm_config.llm_model,
    )
    embeddings = create_embedder(config.embedder)
    transformer_llm = LangchainLLMWrapper(generator_llm)
    ragas_embedding = LangchainEmbeddingsWrapper(embeddings)
    trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=ragas_embedding)
    apply_transforms(kg, trans)
    generator = TestsetGenerator(
        llm=transformer_llm,
        embedding_model=ragas_embedding,
        knowledge_graph=kg,
        persona_list=[Persona(name=person.name, role_description=person.role_description) for person in config.persona_list],
    )
    return generator