from typing import TypeVar, cast

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from bot.llm.client import LLMClient
from bot.logging import log_event

T = TypeVar("T", bound=BaseModel)

class OpenAIClient(LLMClient):
    def __init__(self, chat: BaseChatModel) -> None:
        self._chat = chat

    @classmethod
    def create(
        cls,
        *,
        model: str = "gpt-4.1-mini",
    ) -> "OpenAIClient":
        return cls(
            ChatOpenAI(
                model=model,
            )
        )
    
    def _build_messages(self, *, prompt: str, system_instructions: str | None = None) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        return messages
    
    def generate(self, *, prompt: str, system_instructions: str | None = None) -> str:
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        response: AIMessage = self._chat.invoke(messages)
        return response.text
    
    def generate_with_structured_output(self, *, prompt: str, output_format: type[T], system_instructions: str | None = None) -> T:
        structured_model = self._chat.with_structured_output(output_format)
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
    
    async def agenerate_with_structured_output(self, *, prompt: str, output_format: type[T], system_instructions: str | None = None) -> T:
        structured_model = self._chat.with_structured_output(output_format)
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        result = await structured_model.ainvoke(
            messages
        )

        log_event(
            event="llm_client.agenerate_with_structured_output",
            payload={
                "prompt": prompt,
                "system_instructions": system_instructions,
                "llm_response": result.model_dump() if isinstance(result, BaseModel) else str(result),
            }
        )

        return cast(T, result)
    
    async def agenerate_with_structured_output_batch(
        self,
        *,
        prompts: list[str],
        output_format: type[T],
        system_instructions: str | None = None,
        max_concurrency: int = 10,
    ) -> list[T]:

        structured_model = self._chat.with_structured_output(output_format)

        inputs: list[LanguageModelInput] = [
            self._build_messages(
                prompt=prompt,
                system_instructions=system_instructions,
            )
            for prompt in prompts
        ]

        raw_results = await structured_model.abatch(
            inputs,
            config={
                "max_concurrency": max_concurrency,
            },
        )

        results = [cast(T, result) for result in raw_results]

        log_event(
            event="llm_client.agenerate_with_structured_output_batch",
            payload={
                "prompts": prompts,
                "system_instructions": system_instructions,
                "llm_response": [
                    result.model_dump()
                    for result in results
                ],
            },
        )

        return results