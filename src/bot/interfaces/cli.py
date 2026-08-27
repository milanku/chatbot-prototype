from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from bot.config.prompts_config import PROMPT_CONFIGS
from bot.config.settings import Settings
from bot.engine import ChatbotEngine, EngineAnswerResult, EngineConfig, EngineDeps
from bot.llm.client import LLMClient
from bot.llm.openai_client import OpenAIClient
from bot.logging import setup_logging
from bot.memory.models import SessionStore
from bot.memory.session_store import InMemorySessionStore
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.tx_qa.transactions_repository_mock import (
    TransactionsRepositoryFromJsonMock,
)
from bot.trace_context import bind_session_id, get_current_session_id

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")) -> None:
    setup_logging(verbose=verbose)

    session_id = uuid4().hex
    session_store: SessionStore = InMemorySessionStore()
    settings = Settings()  # Load settings (e.g., API keys) from environment variables or config files
    llm_client: LLMClient = OpenAIClient.create(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_LLM_MODEL,
    )
    embedder: Embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    tx_repository = TransactionsRepositoryFromJsonMock.from_json_file(
        Path(settings.TRANSACTIONS_MOCK_PATH)
    )
    doc_store = DocStore.build_doc_store(
        embedder=embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
    )
    engine_config = EngineConfig()
    engine_deps = EngineDeps(
        tx_repository=tx_repository,
        doc_repository=doc_store,
        embedder=embedder,
        llm_client=llm_client,
        prompt_configs=PROMPT_CONFIGS
    )
    
    engine = ChatbotEngine(engine_config, engine_deps)

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    with bind_session_id(session_id):
        while True:
            msg = typer.prompt("> ")
            if msg.strip().lower() in {"exit", "quit"}:
                break
            
            result: EngineAnswerResult = engine.answer(
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
