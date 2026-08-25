from dataclasses import dataclass, field
from typing import TypeVar, cast

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from bot.llm.client import LLMClient
from bot.logging import log_event

T = TypeVar("T", bound=BaseModel)

@dataclass
class OpenAIClient(LLMClient):
    api_key: SecretStr = field(repr=False)
    default_model: str = "gpt-4.1-mini"
    
    def _get_chat_model(self, model: str) -> BaseChatModel:
        return ChatOpenAI(model=model or self.default_model, api_key=self.api_key)
    
    def _build_messages(self, *, prompt: str, system_instructions: str | None = None) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        return messages
    
    def generate(self, *, prompt: str, system_instructions: str | None = None) -> str:
        chat = self._get_chat_model(model=self.default_model)
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        response: AIMessage = chat.invoke(messages)
        return response.text
    
    def generate_with_structured_output(self, *, prompt: str, output_format: type[T], system_instructions: str | None = None) -> T:
        chat = self._get_chat_model(model=self.default_model)
        structured_model = chat.with_structured_output(output_format)
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        result = structured_model.invoke(
            messages
        )
        
        log_event(
            event="llm_client.generate_with_structured_output",
            payload={
                "prompt": prompt,
                "system_instructions": system_instructions,
                "llm_response": result.model_dump() if isinstance(result, BaseModel) else str(result),
            }
        )
        
        return cast(T, result)