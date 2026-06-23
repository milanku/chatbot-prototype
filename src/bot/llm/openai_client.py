from dataclasses import dataclass, field

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bot.llm.client import LLMClient


@dataclass
class OpenAIClient(LLMClient):
    api_key: str = field(repr=False)
    
    def _get_chat_model(self, model: str = "gpt-4.1-mini") -> ChatOpenAI:
        return ChatOpenAI(model=model, api_key=self.api_key)
    
    def generate(self, prompt: str, system_instructions: str | None = None) -> str:
        chat = self._get_chat_model()
        messages = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        response = chat.invoke(messages)
        return response.content