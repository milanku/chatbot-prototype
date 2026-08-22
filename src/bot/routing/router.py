from bot.llm.client import LLMClient
from bot.models.routing import Route, RouterDecision, RouterDecisionExtraction
from bot.routing.prompt_builder import (
    RouterPromptInput,
    RouterPromptLoader,
)


def select_route(
    *, 
    llm_client: LLMClient,
    message: str,
    prompt_loader: RouterPromptLoader
) -> RouterDecision:
    system_prompt = prompt_loader.build_system_instructions(
        allowed_routes=[route.value for route in Route]
    )
    user_prompt = prompt_loader.build_user_prompt(RouterPromptInput(message=message))
      
    router_response = llm_client.generate_with_structured_output(
        prompt=user_prompt,
        output_format=RouterDecisionExtraction,
        system_instructions=system_prompt,
    )
    
    return RouterDecision(
        route=router_response.decision.route,
    )