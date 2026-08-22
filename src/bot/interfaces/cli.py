from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from bot.config.prompts_config import (
    CLAIM_EXTRACTOR_PROMPT_CONFIG,
    CLAIM_VERIFIER_PROMPT_CONFIG,
    EXPLAIN_PARSE_PROMPT_CONFIG,
    TIMEFRAME_PARSER_PROMPT_CONFIG,
)
from bot.config.settings import Settings
from bot.engine import ChatbotEngine, EngineConfig, EngineDeps
from bot.llm import openai_client
from bot.logging import setup_logging
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
    session_store = InMemorySessionStore()
    settings = Settings()  # Load settings (e.g., API keys) from environment variables or config files
    llm_client = openai_client.OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
    )
    embedder: Embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDINGS_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    tx_repository = TransactionsRepositoryFromJsonMock.from_json_file(
        Path(settings.TRANSACTIONS_MOCK_PATH)
    )
    doc_repository = DocStore.build_doc_store(
        embedder=embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
    )
    engine_config = EngineConfig(
        claim_extractor_config=CLAIM_EXTRACTOR_PROMPT_CONFIG,
        claim_verifier_config=CLAIM_VERIFIER_PROMPT_CONFIG,
        explain_parse_config=EXPLAIN_PARSE_PROMPT_CONFIG,
        timeframe_parser_config=TIMEFRAME_PARSER_PROMPT_CONFIG,
    )
    engine_deps = EngineDeps(
        tx_repository=tx_repository,
        doc_repository=doc_repository,
        llm_client=llm_client,
    )
    
    engine = ChatbotEngine(engine_config, engine_deps)

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
