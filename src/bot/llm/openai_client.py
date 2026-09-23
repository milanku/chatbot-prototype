from typing import Callable, TypeVar, cast

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
    
    def _build_messages(
        self,
        *,
        prompt: str,
        system_instructions: str | None = None
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        return messages
    
    def generate(
        self,
        *,
        prompt: str,
        system_instructions: str | None = None
    ) -> str:
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        response: AIMessage = self._chat.invoke(messages)
        return response.text
    
    def generate_with_structured_output(
        self,
        *,
        prompt: str,
        output_format: type[T],
        system_instructions: str | None = None,
        check_is_output_valid: Callable[[T], bool] | None = None,
        retries: int = 1,
    ) -> T:
        structured_model = self._chat.with_structured_output(output_format)
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        
        # Sometimes the structured output is not enough to guarantee validity
        # Example: Valid JSON structure but some elements of a list might be missing
        for attempt in range(retries):
            try:
                result = structured_model.invoke(
                    messages
                )
                if check_is_output_valid is None or check_is_output_valid(cast(T, result)):
                    break
                
                log_event(
                    event="llm_client.generate_with_structured_output.invalid_output",
                    payload={
                        "prompt": prompt,
                        "system_instructions": system_instructions,
                        "result": str(result) if "result" in locals() else None,
                    }
                )
            except Exception as e:
                log_event(
                    event="llm_client.generate_with_structured_output.error",
                    payload={
                        "prompt": prompt,
                        "system_instructions": system_instructions,
                        "error": str(e),
                    }
                )
                if attempt == retries - 1:
                    raise e
        else:
            raise ValueError("Failed to generate valid output after retries")
        
        log_event(
            event="llm_client.generate_with_structured_output.success",
            payload={
                "prompt": prompt,
                "system_instructions": system_instructions,
                "llm_response": result.model_dump() if isinstance(result, BaseModel) else str(result),
            }
        )
        
        return cast(T, result)
    
    async def agenerate_with_structured_output(
        self,
        *,
        prompt: str,
        output_format: type[T],
        system_instructions: str | None = None,
        check_is_output_valid: Callable[[T], bool] | None = None,
        retries: int = 1,
    ) -> T:
        structured_model = self._chat.with_structured_output(output_format)
        messages: list[BaseMessage] = self._build_messages(prompt=prompt, system_instructions=system_instructions)
        
        # Sometimes the structured output is not enough to guarantee validity
        # Example: Valid JSON structure but some elements of a list might be missing
        for attempt in range(retries):
            try:
                result = await structured_model.ainvoke(
                    messages
                )
                if check_is_output_valid is None or check_is_output_valid(cast(T, result)):
                    break
            except Exception as e:
                if attempt == retries - 1:
                    raise e
        else:
            raise ValueError("Failed to generate valid output after retries")

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
        check_is_output_valid: Callable[[list[T]], bool] | None = None,
        retries: int = 1,
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

        # Sometimes the structured output is not enough to guarantee validity
        # Example: Valid JSON structure but some elements of a list might be missing
        for attempt in range(retries):
            try:
                raw_results = await structured_model.abatch(
                    inputs,
                    config={
                        "max_concurrency": max_concurrency,
                    },
                )
                if check_is_output_valid is None or check_is_output_valid([cast(T, result) for result in raw_results]):
                    break
            except Exception as e:
                if attempt == retries - 1:
                    raise e
        else:
            raise ValueError("Failed to generate valid output after retries")

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