from bot.llm.client import LLMClient
from bot.llm.config import LLMConfig
from bot.llm.openai_client import OpenAIClient


def get_llm(llm_config: LLMConfig) -> LLMClient:
    if llm_config.llm_provider == "openai":
        return OpenAIClient.create(
            model=llm_config.llm_model,
        )
    raise ValueError(f"Unsupported LLM provider: {llm_config.llm_provider}")