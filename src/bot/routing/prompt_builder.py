from dataclasses import dataclass
from pathlib import Path

from bot.models.routing import Route


@dataclass(frozen=True)
class RouterPromptInput:
    message: str
    
ROUTE_DESCRIPTIONS: dict[Route, str] = {
    Route.TX_SUMMARY: (
        'TX_SUMMARY\n'
        'Choose this when the user is asking for a transaction-based spending summary that should be computed from bank transactions.\n'
        'Typical examples:\n'
        '- "How much did I spend on pets in February?"\n'
        '- "What did I spend on food last month?"\n'
        '- "How much did I spend on other in January 2026?"\n'
        '- "Total spending on pets this year"\n'
        'This route applies even if the request is incomplete, for example if the timeframe is missing but the user is clearly asking for a spending total.'
    ),
    Route.TX_LIST: (
        'TX_LIST\n'
        'Choose this when the user is asking for a list of transactions matching a new query without reference to a previous summary.\n'
        'Typical examples:\n'
        '- "List all my transactions from last month"\n'
        '- "Show me my transactions in January 2026"\n'
        '- "What transactions did I have on February 14th?"\n'
        '- "List all my expenses above $100"\n'
        'This route applies even if the request is incomplete, for example if the timeframe is missing but the user is clearly asking for a list of transactions.'
    ),
    Route.TX_EXPLAIN: (
        'TX_EXPLAIN\n'
        'Choose this when the user is asking to see the transactions that make up a previously discussed spending sum.\n'
        'Typical examples:\n'
        '- "Show me the transactions that built up this sum"\n'
        '- "Which transactions are included?"\n'
        '- "List those payments"\n'
        '- "What makes up that amount?"\n'
        'Choose this route even if the previous summary may be missing. The executor will handle that later.'
    ),
    Route.DOCS_ANSWER: (
        'DOCS_ANSWER\n'
        'Choose this when the user is asking an informational question that should be answered from markdown documentation.\n'
        'Typical examples:\n'
        '- "How do refunds work?"\n'
        '- "What is the return policy?"\n'
        '- "How can I reset my password?"\n'
        '- "What are the supported payment methods?"\n'
        'Choose this only when the message looks like a documentation question.'
    ),
    Route.OUT_OF_SCOPE: (
        'OUT_OF_SCOPE\n'
        'Choose this when the message does not fit any of the routes above.\n'
        'Typical examples:\n'
        '- "What is the capital of France?"\n'
        '- "Write me a poem"\n'
        '- "What\'s the weather today?"\n'
        '- anything unrelated to transaction summaries, transaction follow-up explanations, or documentation-based questions'
    ),
}

def load_router_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_router_system_prompt(template: str, allowed_routes: list[str]) -> str:
    allowed_routes_str = "\n\n".join(allowed_routes)
    allowed_routes_descriptions_str = "\n\n".join(ROUTE_DESCRIPTIONS[Route(route)] for route in allowed_routes)
    allowed_routes_list_with_commas = ", ".join(allowed_routes)
    
    return template.format(
        allowed_routes=allowed_routes_str,
        allowed_routes_descriptions=allowed_routes_descriptions_str,
        allowed_routes_list_with_commas=allowed_routes_list_with_commas
    )

def build_router_user_prompt(input: RouterPromptInput) -> str:
    return f"Classify this message.\n\nUser message: {input.message}"