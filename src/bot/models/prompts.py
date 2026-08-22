from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bot.config.prompts_config import PromptConfig

PromptInputT = TypeVar("PromptInputT")
    

class PromptLoader(ABC, Generic[PromptInputT]):
    def __init__(self, *, prompt_config: PromptConfig) -> None:
        self.prompt_config = prompt_config

    def load_system_instructions(self) -> str:
        path = self.prompt_config.instructions_file_path

        if not path.exists():
            raise FileNotFoundError(
                f"Instructions file not found: {path}"
            )

        return path.read_text(encoding="utf-8")

    @abstractmethod
    def build_user_prompt(self, input: PromptInputT) -> str:
        ...