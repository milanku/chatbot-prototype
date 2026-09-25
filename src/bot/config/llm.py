from pydantic import BaseModel


class OpenAILLMConfig(BaseModel):
    model: str


LLMConfig = OpenAILLMConfig
