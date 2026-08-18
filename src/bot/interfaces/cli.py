from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer
from langchain_openai import OpenAIEmbeddings

from bot.config import Settings
from bot.engine import ChatbotEngine, EngineConfig, EngineDeps
from bot.llm import openai_client
from bot.logging import setup_logging
from bot.memory.session_store import InMemorySessionStore
from bot.routes.doc_qa.bootstrap import build_doc_store
from bot.routes.tx_qa.transactions_repository_mock import (
    TransactionsRepositoryFromJsonMock,
)
from bot.trace_context import bind_session_id, get_current_session_id

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")) -> None:
    setup_logging(verbose=verbose)

    session_store = InMemorySessionStore()
    session_id = uuid4().hex
    settings = Settings()  # Load settings (e.g., API keys) from environment variables or config files
    llm_client = openai_client.OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
    )
    embedder = OpenAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    
    tx_repository = TransactionsRepositoryFromJsonMock.from_json_file(
        Path(settings.TRANSACTIONS_MOCK_PATH)
    )
    doc_repository = build_doc_store(
        embedder=embedder,
        chunking_version=settings.CHUNKING_VERSION,
        md_docs_dir=Path(settings.DOCS_PATH),
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
    )
    engine = ChatbotEngine(EngineConfig(), EngineDeps(tx_repository=tx_repository, doc_repository=doc_repository, llm_client=llm_client))

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    with bind_session_id(session_id):
        while True:
            msg = typer.prompt("> ")
            if msg.strip().lower() in {"exit", "quit"}:
                break
            
            response, new_state = engine.answer(
                msg, session_state=session_store.get_session(get_current_session_id() or session_id)
            )
            session_store.set_session(session_id, new_state)  # Update session state
            
            typer.echo(f"\n\n{response.answer}\n\n")
            if(response.doc_references):
                typer.echo(        
                    "Referenced documents:\n" + "\n".join(f"{ref.file_name} ({' >> '.join(ref.heading_path)})" for ref in response.doc_references) + "\n\n"
                )
            typer.echo(f"(trace_id: {response.trace_id})")
