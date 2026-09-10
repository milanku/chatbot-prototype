
import argparse
import json
import re
from pathlib import Path

from FlagEmbedding.inference import FlagReranker
from langchain.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas.testset import Testset

from bot.config.prompts_config import PromptConfig
from bot.config.settings import Settings
from bot.llm.openai_client import OpenAIClient
from bot.logging import setup_logging
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.reranker import CrossEncoderReranker
from bot.routes.doc_qa.retriever import ChunksRetriever, RetrievalConfig
from prompt_evals.docs_qa.LLMJudge import (
    EvaluationChunk,
    LLMJudge,
    RelevanceJudgePromptLoader,
)


def clean_context(context: str) -> str:
    return re.sub(r"^<\d+-hop>\n\n", "", context)

def calc_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    return len(retrieved & relevant) / len(retrieved)

def calc_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)

def create_chunk_union(chunk_groups: list[list[EvaluationChunk]]) -> list[EvaluationChunk]:
    # Creates a union of all evaluation chunks from the given groups, duplicates are removed
    union: list[EvaluationChunk] = []
    seen_chunk_ids: set[str] = set()

    for group in chunk_groups:
        for chunk in group:
            if chunk.chunk_id not in seen_chunk_ids:
                union.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)

    return union

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
    #     model=settings.OPENAI_LLM_MODEL,
    # )
    embedder: Embeddings = JinaEmbeddings()
    qwen_embedder: Embeddings = Qwen3Embeddings()
    doc_store = DocStore.build_doc_store(
        embedder=embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
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
    
    generator_llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_LLM_MODEL,
    )

    testset: Testset = Testset.from_jsonl(Path("prompt_evals/docs_qa/ragas_dataset.jsonl"))
    doc_chunks = doc_store.get_embedded_chunks()
    doc_chunks_by_content = {chunk.content: chunk for chunk in doc_chunks}
    
    pandas = testset.to_pandas()
    
    
    # for sample in testset.samples:
    
    sample = testset.samples[5]
    
    user_question = sample.eval_sample.user_input
    retrieved_contexts = [
        clean_context(ctx)
        for ctx in sample.eval_sample.reference_contexts
    ]
    
    retrieved_contexts_ids = [doc_chunks_by_content[ctx].chunk_id for ctx in retrieved_contexts if ctx in doc_chunks_by_content]
    
    retrieved_docs = retriever.retrieve(question=user_question.__str__())
    reranked_docs = reranker.rerank(user_question.__str__(), retrieved_docs)[:4]
    
    retriever_precision = calc_precision(
        retrieved_ids=[doc.id for doc in retrieved_docs],
        relevant_ids=retrieved_contexts_ids,
    )
    reranker_precision = calc_precision(
        retrieved_ids=[doc.id for doc in reranked_docs],
        relevant_ids=retrieved_contexts_ids,
    )
    
    retriever_recall = calc_recall(
        retrieved_ids=[doc.id for doc in retrieved_docs],
        relevant_ids=retrieved_contexts_ids,
    )
    reranker_recall = calc_recall(
        retrieved_ids=[doc.id for doc in reranked_docs],
        relevant_ids=retrieved_contexts_ids,
    )

    retrieved_docs_scores = [doc.retrieval_score for doc in retrieved_docs]
    reranked_docs_scores = [doc.reranker_score for doc in reranked_docs]

    print(f"Question: {user_question}")
    print(f"Required context: {[f'{ctx}' for ctx in retrieved_contexts]}")
    
    print(f"Retrieved docs: {[f'{doc}' for doc in retrieved_docs]}")
    print(f"Reranked docs: {[f'{doc}' for doc in reranked_docs]}")
    
    print(f"Retriever/reranker precision: {retriever_precision*100}% : {reranker_precision*100}%")
    print(f"Retriever/reranker recall: {retriever_recall*100}% : {reranker_recall*100}%")

    # ***************************************************
    
    evaluation_doc_store = DocStore.build_doc_store(
        embedder=qwen_embedder,
        embedding_model="qwen/Qwen3-Embedding-4B",
        embeddings_dir=Path("prompt_evals/embeddings"),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path("prompt_evals/embeddings/manifest.json"),
        chunking_version=settings.CHUNKING_VERSION,
    )
    
    embedded_chunks = evaluation_doc_store.get_embedded_chunks()
    evaluation_retriever = ChunksRetriever(
        embedder=qwen_embedder,
        doc_store=evaluation_doc_store,
        config=RetrievalConfig(top_k=10),
    )
    
    relevant_evaluation_chunks = evaluation_retriever.retrieve(question=user_question.__str__())
    
    bm25retriever = BM25Retriever.from_documents(
        documents=[
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.chunk_id
                }
            ) for chunk in embedded_chunks
        ]
    )
    bm25retriever.k = 10
    bm25chunks = bm25retriever.invoke(user_question.__str__())
    
    print("--------------------------------------------------")
    print("Relevant chunks according to original retriever")
    for chunk in retrieved_docs:
        print(f"Chunk ID: {chunk.id}, Content: {chunk.content}")
    print("--------------------------------------------------")
    print("Relevant chunks according to evaluation retriever")
    for chunk in relevant_evaluation_chunks:
        print(f"Chunk ID: {chunk.id}, Content: {chunk.content}")
    print("--------------------------------------------------")
    print("Relevant chunks according to BM25 retriever")
    for chunk in bm25chunks:
        print(f"Chunk ID: {chunk.metadata['chunk_id']}, Content: {chunk.page_content}")
    
    chunk_union = create_chunk_union([
            [EvaluationChunk(chunk_id=chunk.id, content=chunk.content) for chunk in retrieved_docs],
            [EvaluationChunk(chunk_id=chunk.id, content=chunk.content) for chunk in relevant_evaluation_chunks],
            [EvaluationChunk(chunk_id=chunk.metadata['chunk_id'], content=chunk.page_content) for chunk in bm25chunks]
    ])
    
    print(len(chunk_union))

    llm_with_different_model = OpenAIClient.create(
        api_key=settings.OPENAI_API_KEY,
        model="gpt-5.4-mini",
    )
    relevance_judge_prompt_config = PromptConfig(
        directory=Path("prompt_evals/docs_qa/instructions/llm_chunk_relevance_judge"),
        version="v001"
    )
    relevance_judge_prompt_loader = RelevanceJudgePromptLoader(prompt_config=relevance_judge_prompt_config)
    llm_judge = LLMJudge(
        llm_client=llm_with_different_model,
        relevance_judge_prompt_loader=relevance_judge_prompt_loader,
    )
    judge_result = llm_judge.judge(
        user_question=user_question.__str__(),
        chunk_pool=chunk_union,
    )
    print(f"Judge result: {judge_result}")
    
    Path("prompt_evals/docs_qa/llm_chunk_relevance_judge.json").write_text(
        json.dumps([node.model_dump() for node in judge_result], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )



if __name__ == "__main__":
    main()