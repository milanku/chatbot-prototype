from bot.llm.client import LLMClient
from bot.llm.openai_client import OpenAIClient


def get_llm(llm_model: str) -> LLMClient:
    return OpenAIClient.create(
        model=llm_model,
    )