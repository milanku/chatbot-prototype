from dataclasses import dataclass

from bot.handlers.models import RouteHandlerResult


@dataclass(frozen=True)
class EngineResult:
    route_result: RouteHandlerResult
    trace_id: str
