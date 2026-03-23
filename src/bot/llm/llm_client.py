from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMClient(Protocol):
        
    def generate(self, prompt: str) -> str: ...