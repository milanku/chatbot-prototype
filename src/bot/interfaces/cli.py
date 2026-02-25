from __future__ import annotations

from pathlib import Path

import typer

from bot.engine import ChatbotEngine, EngineConfig, EngineDeps
from bot.logging import setup_logging
from bot.recipes.tx_qa.mock_repository import JsonMockTransactionsRepository

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main() -> None:
    setup_logging()

    tx_repository = JsonMockTransactionsRepository.from_json_file(
        Path("data/mocks/transactions_mock.json")
    )
    engine = ChatbotEngine(EngineConfig(), EngineDeps(tx_repository=tx_repository))

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    while True:
        msg = typer.prompt("> ")
        if msg.strip().lower() in {"exit", "quit"}:
            break
        response = engine.answer(msg)
        typer.echo(response.answer)
        typer.echo(f"(trace_id: {response.trace_id})")
        typer.echo(f"(trace_id: {response.trace_id})")
