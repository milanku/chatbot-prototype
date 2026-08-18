from dataclasses import dataclass
from pathlib import Path

from bot.models.routing import Route


@dataclass(frozen=True)
class RouterPromptInput:
    message: str
    
ROUTE_DESCRIPTIONS_PATHS: dict[Route, Path] = {
    Route.TX_SUMMARY: Path("src/bot/prompts/router_path_descriptions/tx_summary_description.txt"),
    Route.TX_LIST: Path("src/bot/prompts/router_path_descriptions/tx_list_description.txt"),
    Route.TX_EXPLAIN: Path("src/bot/prompts/router_path_descriptions/tx_explain_description.txt"),
    Route.DOCS_ANSWER: Path("src/bot/prompts/router_path_descriptions/docs_answer_description.txt"),
    Route.OUT_OF_SCOPE: Path("src/bot/prompts/router_path_descriptions/out_of_scope_description.txt"),
}

def load_router_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_router_system_prompt(template: str, allowed_routes: list[str]) -> str:
    allowed_routes_str = "\n\n".join(allowed_routes)
    allowed_routes_descriptions_str = "\n\n".join(load_router_instructions(ROUTE_DESCRIPTIONS_PATHS[Route(route)]) for route in allowed_routes)
    allowed_routes_list_with_commas = ", ".join(allowed_routes)
    
    return template.format(
        allowed_routes=allowed_routes_str,
        allowed_routes_descriptions=allowed_routes_descriptions_str,
        allowed_routes_list_with_commas=allowed_routes_list_with_commas
    )

def build_router_user_prompt(input: RouterPromptInput) -> str:
    return f"Classify this message.\n\nUser message: {input.message}"