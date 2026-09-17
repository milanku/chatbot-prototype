import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import apply_transforms, default_transforms

from bot.config.settings import Settings
from bot.logging import setup_logging
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.local_embeddings import JinaEmbeddings


def main() -> None:
    print("Script started")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    settings = Settings()  # Load settings (e.g., API keys) from environment variables or config files
    
    # llm_client: LLMClient = OpenAIClient.create(
    #     api_key=settings.OPENAI_API_KEY,
    #     model=settings.DEFAULT_OPENAI_LLM_MODEL,
    # )
    embedder: Embeddings = JinaEmbeddings()
    doc_store = DocStore.build_doc_store(
        embedder=embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
    )
    
    generator_llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.DEFAULT_OPENAI_LLM_MODEL,
    )
    
    loan_chunks = [chunk for chunk in doc_store.get_embedded_chunks() if "05_loans_overdrafts_and_credit_assessment.md" in chunk.doc_reference.file_name]
    
    kg = KnowledgeGraph()
    
    for chunk in doc_store.get_embedded_chunks():
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
    
    
    docs: list[Document] = [
        Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.doc_reference.file_name,
            }
        ) for chunk in doc_store.get_embedded_chunks()
    ]
    
    transformer_llm = LangchainLLMWrapper(generator_llm)
    ragas_embedding = LangchainEmbeddingsWrapper(embedder)

    #query_distribution = default_query_distribution(transformer_llm)
    trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=ragas_embedding)
    apply_transforms(kg, trans)
    
    for node in kg.nodes:
        print("TYPE:", node.type)
        print("PROPERTIES:", node.properties)
        print("---")
        
    persona_list = [
        Persona(
            name="Everyday retail customer",
            role_description=(
                "A retail banking customer who uses Aurora Bank for everyday "
                "banking and asks practical questions about products, accounts, "
                "payments, fees, security, and customer service."
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
            name="Existing customer needing support",
            role_description=(
                "An Aurora Bank customer trying to resolve a banking problem or "
                "understand what to do next. They may ask about failed "
                "transactions, disputes, fraud, account access, complaints, "
                "security, fees, or changes to their account."
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

    generator = TestsetGenerator(
        llm=transformer_llm,
        embedding_model=ragas_embedding,
        knowledge_graph=kg,
        persona_list=persona_list,
    )
    dataset = generator.generate(testset_size=200, num_personas=len(persona_list))
    
    for sample in dataset.samples:
        print(sample)
        print(vars(sample.eval_sample))
        break
    
    dataset.to_jsonl("ragas_dataset.jsonl")
    # dataset.to_pandas()
    # 
    # print("Dataset generation completed")
    dataset.to_csv("ragas_dataset.csv")
    # print("Dataset saved to ragas_dataset.csv")
    
    
    
if __name__ == "__main__":
    main()