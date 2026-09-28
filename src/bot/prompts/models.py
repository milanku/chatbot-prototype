from abc import ABC, abstractmethod
from importlib.resources import files
from typing import Generic, TypeVar

from bot.config.prompts_config import PromptConfig

PromptInputT = TypeVar("PromptInputT")


class PromptLoader(ABC, Generic[PromptInputT]):
    def __init__(self, *, prompt_config: PromptConfig) -> None:
        self.prompt_config = prompt_config

    def load_system_instructions(self) -> str:
        return (
            files(self.prompt_config.package)
            .joinpath(self.prompt_config.instructions_file_path)
            .read_text(encoding="utf-8")
        )

    @abstractmethod
    def build_user_prompt(self, input: PromptInputT) -> str: ...
