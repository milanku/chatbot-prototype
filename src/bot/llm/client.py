from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

@dataclass
class LLMClient():
        
    def generate(self, *, prompt: str, system_instructions: str | None = None) -> str: ...
    
    def generate_with_structured_output(self, *, prompt: str, output_format: type[T], system_instructions: str | None = None) -> T: ...