from bot.llm.client import LLMClient
from bot.models.routing import Route, RouterDecision, RouterDecisionExtraction
from bot.routing.router_prompt_loader import (
    RouterPromptInput,
    RouterPromptLoader,
)


class RouteSelector:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_loader: RouterPromptLoader
    ) -> None:
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader

    def select(
        self,
        *,
        message: str
    ) -> RouterDecision:
        system_prompt = self._prompt_loader.build_system_instructions(
            allowed_routes=[route.value for route in Route]
        )
        user_prompt = self._prompt_loader.build_user_prompt(RouterPromptInput(message=message)) 
        router_response = self._llm_client.generate_with_structured_output(
            prompt=user_prompt,
            output_format=RouterDecisionExtraction,
            system_instructions=system_prompt,
        )
        
        return RouterDecision(
            route=router_response.decision.route,
        )