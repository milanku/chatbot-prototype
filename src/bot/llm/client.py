from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMClient(Protocol):
    def generate(self, *, prompt: str, system_instructions: str | None = None) -> str: ...

    def generate_with_structured_output(
        self,
        *,
        prompt: str,
        output_format: type[T],
        system_instructions: str | None = None,
        check_is_output_valid: Callable[[T], bool] | None = None,
        retries: int = 1,
    ) -> T: ...

    async def agenerate_with_structured_output(
        self,
        *,
        prompt: str,
        output_format: type[T],
        system_instructions: str | None = None,
        check_is_output_valid: Callable[[T], bool] | None = None,
        retries: int = 1,
    ) -> T: ...

    async def agenerate_with_structured_output_batch(
        self,
        *,
        prompts: list[str],
        output_format: type[T],
        system_instructions: str | None = None,
        check_is_output_valid: Callable[[list[T]], bool] | None = None,
        retries: int = 1,
    ) -> list[T]: ...
