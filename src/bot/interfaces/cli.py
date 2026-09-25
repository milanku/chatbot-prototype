from __future__ import annotations

from datetime import datetime

import typer
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from bot.config.bot import BOT_CONFIG
from bot.config.prompts_config import PROMPT_CONFIGS
from bot.doc_qa.indexing.chunkers.factory import create_chunker
from bot.doc_qa.indexing.embeddings_store_factory import EmbeddingsStoreFactory
from bot.doc_qa.indexing.models import Chunker
from bot.doc_qa.indexing.store_persistor import EmbeddingsStoreRepository
from bot.doc_qa.retrieval.embedders.factory import create_embedder
from bot.engine import ChatbotEngine, EngineDeps
from bot.engine_models import EngineResponse
from bot.llm.client import LLMClient
from bot.llm.factory import create_llm
from bot.logging import generate_id, setup_logging
from bot.trace_context import bind_session_id, get_current_session_id
from bot.tx_qa.indexing.transactions_repository import (
    TransactionsRepositoryFromJsonMock,
)
from bot.tx_qa.memory.models import SessionStore
from bot.tx_qa.memory.session_store import InMemorySessionStore

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    load_dotenv()

    current_time = datetime.now()
    session_id = generate_id(current_time)
    session_store: SessionStore = InMemorySessionStore()
    setup_logging(verbose=verbose)

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
            embeddings_dir_path=BOT_CONFIG.embeddings_dir_path,
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
            if new_state is not None:
                session_store.set_session(session_id, new_state)  # Update session state

            typer.echo(f"\n\n{response.answer}\n\n")
            if response.doc_references:
                typer.echo(
                    "Referenced documents:\n"
                    + "\n".join(
                        f"{ref.file_name} ({' >> '.join(ref.heading_path)})"
                        for ref in response.doc_references
                    )
                    + "\n\n"
                )
            typer.echo(f"(trace_id: {response.trace_id})")
