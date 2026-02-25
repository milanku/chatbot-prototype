from __future__ import annotations

import typer

from bot.logging import setup_logging

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main() -> None:
    setup_logging()
    typer.echo("Chatbot prototype (type 'exit' to quit)")

    while True:
        msg = typer.prompt("> ")
        if msg.strip().lower() in {"exit", "quit"}:
            break
        typer.echo("Not implemented yet.")
        # TODO: process the message