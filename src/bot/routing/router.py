import json
from pathlib import Path

from bot.llm.llm_client import LLMClient
from bot.logging import log_event
from bot.models.routing import Route, RouterDecision
from bot.routing.prompt_builder import (
    RouterPromptInput,
    build_router_system_prompt,
    build_router_user_prompt,
    load_router_instructions,
)


def route(session_id: str, llm_client: LLMClient, message: str) -> RouterDecision:
    prompt_template = load_router_instructions(Path("src/bot/routing/router_instructions.txt"))
    system_prompt = build_router_system_prompt(
        template=prompt_template,
        allowed_routes=(Route.TX_SUMMARY, Route.TX_EXPLAIN, Route.DOCS_ANSWER, Route.OUT_OF_SCOPE),
    )
    user_prompt = build_router_user_prompt(RouterPromptInput(message=message))
    
    log_event(
        trace_id=session_id,
        event="router.input",
        payload={"message": message}
    )
    
    router_response = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt,
    )
    
    log_event(
        trace_id=session_id,
        event="router.output",
        payload={"router_response": router_response}
    )
    
    try:
        parsed_response = json.loads(router_response)
        route_value = parsed_response.get("route", "").strip()
        confidence_value = float(parsed_response.get("confidence", 0))
        reason = parsed_response.get("reason", None)
        if route_value not in [route.value for route in Route]:
            return RouterDecision(route=Route.OUT_OF_SCOPE, confidence=0.5, reason="Invalid route in response")
        return RouterDecision(route=Route(route_value), confidence=confidence_value, reason=reason)
        
    except json.JSONDecodeError:
        # If the response is not valid JSON, classify as OUT_OF_SCOPE with low confidence
        return RouterDecision(route=Route.OUT_OF_SCOPE, confidence=0.5, reason="Invalid JSON response")