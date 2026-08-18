from pathlib import Path

from bot.llm.client import LLMClient
from bot.logging import log_event
from bot.models.routing import Route, RouterDecision, RouterDecisionExtraction
from bot.routing.prompt_builder import (
    RouterPromptInput,
    build_router_system_prompt,
    build_router_user_prompt,
    load_router_instructions,
)


def route(llm_client: LLMClient, message: str) -> RouterDecision:
    prompt_template = load_router_instructions(Path("src/bot/prompts/router_instructions.txt"))
    system_prompt = build_router_system_prompt(
        template=prompt_template,
        allowed_routes=[route.value for route in Route]
    )
    user_prompt = build_router_user_prompt(RouterPromptInput(message=message))
    
    log_event(
        event="router.input",
        payload={"message": message}
    )
    
    router_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=RouterDecisionExtraction,
        system_instructions=system_prompt,
    )
    
    log_event(
        event="router.output",
        payload={"router_response": router_response}
    )
    
    return RouterDecision(
        route=router_response.decision.route,
    )