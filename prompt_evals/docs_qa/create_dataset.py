import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from FlagEmbedding.inference import FlagReranker
from langchain.embeddings import Embeddings
from pydantic import BaseModel

from bot.config.prompts_config import PromptConfig
from bot.config.settings import Settings
from bot.llm.client import LLMClient
from bot.llm.openai_client import OpenAIClient
from bot.logging import log_event, setup_logging
from bot.models.doc_qa.chunks import DocChunkWithEmbedding
from bot.models.prompts import PromptLoader
from bot.routes.doc_qa.coordinator import DocsAnswerCoordinator
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.local_embeddings import JinaEmbeddings
from bot.routes.doc_qa.reranker import CrossEncoderReranker
from bot.routes.doc_qa.retriever import ChunksRetriever, RetrievalConfig
from bot.routes.doc_qa.score import vectors_cosine_similarity


@dataclass(frozen=True)
class KnowledgeBaseInput:
    chunks: list[DocChunkWithEmbedding]
    
class AtomicKnowledgePromptLoader(PromptLoader[KnowledgeBaseInput]):
    def build_user_prompt(self, input: KnowledgeBaseInput) -> str:
        chunks = "\n\n".join(f"Chunk ID: [{input_chunk.chunk_id}]\nContent: {input_chunk.content}" for input_chunk in input.chunks)
        
        return (
            "Generate possible questions from the following chunks.\n\n"
            f"Chunks:\n\n{chunks}\n"
        )

class RequiredChunk(BaseModel):
    chunk_id: str
    contribution: str

class DocChunkQuestions(BaseModel):
    question: str
    required_chunk_contributions: list[RequiredChunk]
    answer: str
    
class DocChunkQuestionsReturnType(BaseModel):
    questions: list[DocChunkQuestions]

class QuestionsGenerator:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        atomic_knowledge_retriever_prompt_loader: AtomicKnowledgePromptLoader
    ):
        self.atomic_knowledge_retriever_prompt_loader = atomic_knowledge_retriever_prompt_loader
        self._llm_client = llm_client
        
    def generate_questions(
            self,
            *, 
            chunks: list[DocChunkWithEmbedding],
        ) -> list[DocChunkQuestions]:
            """
            Generate questions from given chunks.
    
            Args:
                chunks (list[EmbeddedDocChunk]): The chunks from which to generate questions.
    
            Returns:
                list[DocChunkQuestions]: A list of extracted questions for each document chunk.
            """
    
            system_prompt = self.atomic_knowledge_retriever_prompt_loader.load_system_instructions()
            user_prompt = self.atomic_knowledge_retriever_prompt_loader.build_user_prompt(KnowledgeBaseInput(chunks=chunks))
            
            llm_structured_response = self._llm_client.generate_with_structured_output(
                prompt=user_prompt,
                output_format=DocChunkQuestionsReturnType,
                system_instructions=system_prompt,
            )
            return llm_structured_response.questions

class Node(BaseModel):
    chunk_id: str
    content: str
    score: float


class NodeNeighbors(BaseModel):
    chunk_id: str
    content: str
    neighbors: list[Node]  # List of neighboring chunk IDs

def build_neighbor_matrix(chunks: list[DocChunkWithEmbedding], top_k: int) -> list[NodeNeighbors]:
    ids = [chunk.chunk_id for chunk in chunks]
    num_chunks = len(chunks)
    embeddings = [chunk.embedding for chunk in chunks]
    neighbor_matrix = [[0.0 for _ in range(num_chunks)] for _ in range(num_chunks)]
    
    for i in range(num_chunks):
        for j in range(num_chunks):
            if i != j:
                neighbor_matrix[i][j] = vectors_cosine_similarity(embeddings[i], embeddings[j])
    
    node_neighbors_list: list[NodeNeighbors] = []
    for i in range(num_chunks):
        neighbors = [
            Node(chunk_id=ids[j], content=chunks[j].content, score=neighbor_matrix[i][j])
            for j in sorted(range(num_chunks), key=lambda x: neighbor_matrix[i][x], reverse=True)[:top_k]
        ]
        node_neighbors_list.append(NodeNeighbors(chunk_id=ids[i], content=chunks[i].content, neighbors=neighbors))
    
    return node_neighbors_list
    
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
    
    llm_client: LLMClient = OpenAIClient.create(
        api_key=settings.OPENAI_API_KEY,
        model=settings.DEFAULT_OPENAI_LLM_MODEL,
    )
    embedder: Embeddings = JinaEmbeddings()
    doc_store = DocStore.build_doc_store(
        embedder=embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
    )
    
    loan_chunks = [chunk for chunk in doc_store.get_embedded_chunks() if "05_loans_overdrafts_and_credit_assessment.md" in chunk.doc_reference.file_name]
    log_event(
        event="loan_chunks_retrieved",
        payload={
            "num_chunks": len(loan_chunks),
            "chunk_contents": [chunk.content for chunk in loan_chunks]
        }
    )
    
   
    
    questions_json_raw = json.loads(Path("prompt_evals/docs_qa/demo_questions.json").read_text(encoding="utf-8"))
    
    questions = [
        DocChunkQuestions.model_validate(item)
        for item in questions_json_raw
    ]
    log_event(
        event="json_load",
        payload={
            "num_chunks": len(questions)
        }
    )
    
    
    
    node_neighbors_list = build_neighbor_matrix(doc_store.get_embedded_chunks(), top_k=5)
    log_event(
        event="node_neighbors_built",
        payload={
            "num_chunks": len(node_neighbors_list),
            "node_neighbors": [f"{node_neighbor.chunk_id}\nContent: {node_neighbor.content}" for node_neighbor in node_neighbors_list]
        }
    )
    # List neighbors that are from different documents
    different_doc_neighbors = [
        node_neighbor
        for node_neighbor in node_neighbors_list
        if any(neighbor.chunk_id.split("_")[0] != node_neighbor.chunk_id.split("_")[0] for neighbor in node_neighbor.neighbors)
    ]
    Path("prompt_evals/docs_qa/node_neighbors.json").write_text(
        json.dumps([node.model_dump() for node in different_doc_neighbors], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for node_neighbor in different_doc_neighbors:
        log_event(
            event="different_doc_neighbors",    
            payload={
                "num_chunks": len(different_doc_neighbors),
                "chunk_id": node_neighbor.chunk_id,
                "node_neighbors": [f"{neighbor.chunk_id}\nContent: {neighbor.content}" for neighbor in node_neighbor.neighbors]
            }
        )
        
    
    loan_chunk_id = "05_loans_overdrafts_and_credit_assessment.md_5"
    embedded_chunks_as_dict = {chunk.chunk_id: chunk for chunk in doc_store.get_embedded_chunks()}
    
    loan_embedded_doc_chunk = embedded_chunks_as_dict.get(loan_chunk_id)
    
    
    if loan_embedded_doc_chunk is None:
        raise ValueError(f"Loan embedded doc chunk with ID {loan_chunk_id} not found.")
    
    different_doc_neighbors_for_loan_chunk: NodeNeighbors | None = next(
        (node_neighbor for node_neighbor in different_doc_neighbors if node_neighbor.chunk_id == loan_chunk_id),
        None
    )
    
    if different_doc_neighbors_for_loan_chunk is None:
        raise ValueError(f"Different doc neighbors for loan chunk with ID {loan_chunk_id} not found.")
    
    neighbor_embedded_chunks = [
        embedded_chunks_as_dict.get(neighbor.chunk_id)
        for neighbor in different_doc_neighbors_for_loan_chunk.neighbors
    ]
    
    non_null_neighbor_embedded_chunks = [chunk for chunk in neighbor_embedded_chunks if chunk is not None]
    
    
    prompt_config: PromptConfig = PromptConfig(
        directory=Path("prompt_evals/docs_qa/instructions/questions_generator"),
        version="v001"
    )
    
    retriever = QuestionsGenerator(
        llm_client=llm_client,
        atomic_knowledge_retriever_prompt_loader=AtomicKnowledgePromptLoader(prompt_config=prompt_config)
    )
    # questions = retriever.generate_questions(chunks=[loan_embedded_doc_chunk, *non_null_neighbor_embedded_chunks])
    
    # Path("prompt_evals/docs_qa/demo_questions.json").write_text(
    #         json.dumps([q.model_dump() for q in questions], ensure_ascii=False, indent=2),
    #         encoding="utf-8",
    #     )
    log_event(
        event="questions_generated",
        payload={
            "num_questions": len(questions),
            "questions": [q.model_dump() for q in questions]
        }
    )
    
    retriever = ChunksRetriever(
        embedder=embedder,
        doc_store=doc_store,
        config=RetrievalConfig(top_k=10),
    )
        
    reranker = CrossEncoderReranker(
        reranker=FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=True,
        )
    )
    
    for question in questions:
        doc_hits = retriever.retrieve(question=question.question)
        reranked_doc_hits = reranker.rerank(question.question, doc_hits)[:5]
        
        log_event(
            event="reranked_doc_hits",
            payload={
                "question": question,
                "num_reranked_doc_hits": len(reranked_doc_hits),
                "doc_hits": [f"{doc_hit.id}:{doc_hit.retrieval_score}" for doc_hit in doc_hits],
                "reranked_doc_hits": [f"{doc_hit.id}:{doc_hit.reranker_score}" for doc_hit in reranked_doc_hits],
            }
        )
        
        if(
            all(
                any(doc_hit.id == required_chunk.chunk_id
                    for doc_hit in reranked_doc_hits
                )
                for required_chunk in question.required_chunk_contributions
            )
        ):
            log_event(
                event="all_relevant_doc_hits_found",
                payload={
                    "question": question,
                    "num_reranked_doc_hits": len(reranked_doc_hits),
                    "required": [rc.model_dump() for rc in question.required_chunk_contributions],
                    "reranked_doc_hits": [f"{doc_hit.id}:{doc_hit.reranker_score}" for doc_hit in reranked_doc_hits],
                }
            )
        else:
            log_event(
                event="not_all_relevant_doc_hits_found",
                payload={
                    "question": question,
                    "num_reranked_doc_hits": len(reranked_doc_hits),
                    "required": [rc.model_dump() for rc in question.required_chunk_contributions],
                    "reranked_doc_hits": [f"{doc_hit.id}:{doc_hit.reranker_score}" for doc_hit in reranked_doc_hits],
                }
            )
#        log_event(
#            event="reranked_doc_hits",
#            payload={
#                "question": question,
#                "chunk_id": chunk.chunk_id,
#                "reranked_doc_hits": [f"{doc_hit.doc_reference.file_name}:{doc_hit.id}:{doc_hit.reranker_score}" for doc_hit in reranked_doc_hits]
#            }
#        )

    # Check if all required relevant doc hits are present in the reranked doc hits
        
    
    
    
    
if __name__ == "__main__":
    main()