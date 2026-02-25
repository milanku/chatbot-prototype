from __future__ import annotations

import typer

from bot.engine import ChatbotEngine, EngineConfig
from bot.logging import setup_logging

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main() -> None:
    setup_logging()
    
    engine = ChatbotEngine(EngineConfig())
    
    typer.echo("Chatbot prototype (type 'exit' to quit)")

    while True:
        msg = typer.prompt("> ")
        if msg.strip().lower() in {"exit", "quit"}:
            break
        response = engine.answer(msg)
        typer.echo(response.answer)
        typer.echo(f"(trace_id: {response.trace_id})")