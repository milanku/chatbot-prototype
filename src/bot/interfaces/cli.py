from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from bot.engine import ChatbotEngine, EngineConfig, EngineDeps
from bot.logging import setup_logging
from bot.memory.session_store import InMemorySessionStore
from bot.recipes.doc_qa.mock_store import MockBankDocStore
from bot.recipes.tx_qa.mock_repository import JsonMockTransactionsRepository

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")) -> None:
    setup_logging(verbose=verbose)

    session_store = InMemorySessionStore()
    session_id = uuid4().hex

    tx_repository = JsonMockTransactionsRepository.from_json_file(
        Path("data/mocks/transactions_mock.json")
    )
    doc_repository = MockBankDocStore.from_files(
        [
            Path("data/docs/accounts-and-access.md"),
            Path("data/docs/cards-and-payments.md"),
            Path("data/docs/digital-banking-and-support.md"),
            Path("data/docs/disputes-and-chargebacks.md"),
            Path("data/docs/fees-and-pricing.md"),
            Path("data/docs/loans-and-credit.md"),
            Path("data/docs/privacy-and-data.md")
        ]
    )
    engine = ChatbotEngine(EngineConfig(), EngineDeps(tx_repository=tx_repository, doc_repository=doc_repository))

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    while True:
        msg = typer.prompt("> ")
        if msg.strip().lower() in {"exit", "quit"}:
            break
        response, new_state = engine.answer(
            msg, session_id=session_id, session_state=session_store.get_session(session_id)
        )
        session_store.set_session(session_id, new_state)  # Update session state
        typer.echo(response.answer)
        typer.echo(f"(trace_id: {response.trace_id})")
