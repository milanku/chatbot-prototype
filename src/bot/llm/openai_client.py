from collections.abc import Callable
from typing import TypeVar, cast

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, HumanMessage, SystemMessage
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
        self, prompt: str, system_instructions: str | None = None
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if system_instructions:
            messages.append(SystemMessage(content=system_instructions))
        messages.append(HumanMessage(content=prompt))
        return messages

    def generate(self, *, prompt: str, system_instructions: str | None = None) -> str:
        messages: list[BaseMessage] = self._build_messages(prompt, system_instructions)
        response: AIMessage = self._chat.invoke(messages)

        log_event(
            event="llm_client.generate",
            payload={
                "messages": [m.model_dump() for m in messages],
                "llm_response": response.model_dump(),
            },
        )

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
        messages: list[BaseMessage] = self._build_messages(prompt, system_instructions)

        # Sometimes the structured output is not enough to guarantee validity
        # Example: Valid JSON structure but some elements of a list might be missing
        # Therefore check_is_output_valid can be used for additional criteria
        for attempt in range(retries):
            try:
                response = structured_model.invoke(messages)
                log_event(
                    event="llm_client.generate_with_structured_output.response",
                    payload={
                        "messages": [m.model_dump() for m in messages],
                        "llm_response": response.model_dump()
                        if isinstance(response, BaseModel)
                        else str(response),
                    },
                )

                if check_is_output_valid is None or check_is_output_valid(cast(T, response)):
                    break

            except Exception as e:
                log_event(
                    event="llm_client.generate_with_structured_output.error",
                    payload={
                        "prompt": prompt,
                        "system_instructions": system_instructions,
                        "error": str(e),
                    },
                )
                if attempt == retries - 1:
                    raise e
        else:
            raise ValueError("Failed to generate valid output after retries")

        return cast(T, response)

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
        messages: list[BaseMessage] = self._build_messages(prompt, system_instructions)

        # Sometimes the structured output is not enough to guarantee validity
        # Example: Valid JSON structure but some elements of a list might be missing
        # Therefore check_is_output_valid can be used for additional criteria
        for attempt in range(retries):
            try:
                response = await structured_model.ainvoke(messages)

                log_event(
                    event="llm_client.agenerate_with_structured_output.response",
                    payload={
                        "messages": [m.model_dump() for m in messages],
                        "llm_response": response.model_dump()
                        if isinstance(response, BaseModel)
                        else str(response),
                    },
                )

                if check_is_output_valid is None or check_is_output_valid(cast(T, response)):
                    break

            except Exception as e:
                log_event(
                    event="llm_client.agenerate_with_structured_output.error",
                    payload={
                        "prompt": prompt,
                        "system_instructions": system_instructions,
                        "error": str(e),
                    },
                )
                if attempt == retries - 1:
                    raise e
        else:
            raise ValueError("Failed to generate valid output after retries")

        return cast(T, response)
