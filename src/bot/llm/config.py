from pydantic import BaseModel


class LLMConfig(BaseModel):
    llm_provider: str
    llm_model: str