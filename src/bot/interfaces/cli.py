from __future__ import annotations

from datetime import datetime

import typer
from dotenv import load_dotenv

from bot.composition.engine import create_chatbot_engine
from bot.config.bot import BOT_CONFIG
from bot.config.prompts_config import PROMPT_CONFIGS
from bot.engine_models import EngineResponse
from bot.logging import generate_id, setup_logging
from bot.trace_context import bind_session_id, get_current_session_id
from bot.tx_qa.memory.models import SessionStore
from bot.tx_qa.memory.session_store import InMemorySessionStore

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    load_dotenv()
    setup_logging(verbose=verbose)

    current_time = datetime.now()
    session_id = generate_id(current_time)
    session_store: SessionStore = InMemorySessionStore()

    engine = create_chatbot_engine(
        config=BOT_CONFIG,
        prompt_configs=PROMPT_CONFIGS,
    )

    typer.echo("Chatbot prototype (type 'exit' to quit)")

    with bind_session_id(session_id):
        while True:
            msg = typer.prompt("> ")
            if msg.strip().lower() in {"exit", "quit"}:
                break

            result: EngineResponse = engine.answer(
                msg, session_state=session_store.get_session(get_current_session_id() or session_id)
            )
            trace_id = result.trace_id
            route_result = result.route_result
            new_state = route_result.new_state

            if new_state is not None:
                session_store.set_session(session_id, new_state)  # Update session state

            typer.echo(f"\n\n{route_result.result_to_str()}\n\n")
            typer.echo(f"(trace_id: {trace_id})")
