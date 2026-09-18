from typing import assert_never

from bot.config.llm import LLMConfig, OpenAILLMConfig
from bot.llm.client import LLMClient
from bot.llm.openai_client import OpenAIClient


def create_llm(
    config: LLMConfig,
) -> LLMClient:
    match config:
        case OpenAILLMConfig():
            return OpenAIClient.create(
                model=config.model,
            )

        case _:
            assert_never(config)