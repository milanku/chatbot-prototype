from dataclasses import dataclass, field

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from bot.llm.client import LLMClient


@dataclass
class OpenAIClient(LLMClient):
    api_key: SecretStr = field(repr=False)
    
    def _get_chat_model(self, model: str = "gpt-4.1-mini") -> ChatOpenAI:
        return ChatOpenAI(model=model, api_key=self.api_key)
    
    def generate(self, prompt: str, system_instructions: str | None = None) -> str:
        chat = self._get_chat_model()
        messages: list[BaseMessage] = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        response: AIMessage = chat.invoke(messages)
        
        return response.text