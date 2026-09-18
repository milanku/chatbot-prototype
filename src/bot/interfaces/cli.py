from __future__ import annotations

from uuid import uuid4

import typer
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from bot.chunker.model import Chunker
from bot.config.bot import BOT_CONFIG
from bot.config.prompts_config import PROMPT_CONFIGS
from bot.engine import ChatbotEngine, EngineDeps
from bot.engine_model import EngineResponse
from bot.factories.chunker import create_chunker
from bot.factories.embedder import create_embedder
from bot.factories.llm import create_llm
from bot.llm.client import LLMClient
from bot.logging import setup_logging
from bot.memory.models import SessionStore
from bot.memory.session_store import InMemorySessionStore
from bot.routes.doc_qa.embeddings_store_factory import EmbeddingsStoreFactory
from bot.routes.doc_qa.store_persistor import EmbeddingsStoreRepository
from bot.routes.tx_qa.transactions_repository_mock import (
    TransactionsRepositoryFromJsonMock,
)
from bot.trace_context import bind_session_id, get_current_session_id

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")) -> None:
    setup_logging(verbose=verbose)
    load_dotenv()
    
    session_id = uuid4().hex
    session_store: SessionStore = InMemorySessionStore()
    
    llm_client: LLMClient = create_llm(BOT_CONFIG.llm)
    embedder: Embeddings = create_embedder(BOT_CONFIG.embedder)
    chunker: Chunker = create_chunker(BOT_CONFIG.chunker)
    tx_repository = TransactionsRepositoryFromJsonMock.from_json_file(
        BOT_CONFIG.transactions_mock_file_path
    )
    embeddings_store = EmbeddingsStoreFactory(
        md_docs_dir=BOT_CONFIG.docs_dir_path,
        chunker=chunker,
        chunking_version=BOT_CONFIG.chunker.version,
        embedder=embedder,
        embeddings_model=BOT_CONFIG.embedder.model,
        repository=EmbeddingsStoreRepository(
            embeddings_dir_path=BOT_CONFIG.embeddings_dir_path
        ),
    ).load_or_create()
    
    engine_deps = EngineDeps(
        tx_repository=tx_repository,
        embeddings_store=embeddings_store,
        embedder=embedder,
        llm_client=llm_client,
        prompt_configs=PROMPT_CONFIGS,
        retriever_configs=BOT_CONFIG.retrievers,
        reranker_config=BOT_CONFIG.reranker,
    )
    
    engine = ChatbotEngine(engine_deps)

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    with bind_session_id(session_id):
        while True:
            msg = typer.prompt("> ")
            if msg.strip().lower() in {"exit", "quit"}:
                break
            
            result: EngineResponse = engine.answer(
                msg, session_state=session_store.get_session(get_current_session_id() or session_id)
            )
            response = result.response
            new_state = result.new_state
            if(new_state is not None):
                session_store.set_session(session_id, new_state)  # Update session state
            
            typer.echo(f"\n\n{response.answer}\n\n")
            if(response.doc_references):
                typer.echo(        
                    "Referenced documents:\n" + "\n".join(f"{ref.file_name} ({' >> '.join(ref.heading_path)})" for ref in response.doc_references) + "\n\n"
                )
            typer.echo(f"(trace_id: {response.trace_id})")
