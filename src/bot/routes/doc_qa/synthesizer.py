
from bot.llm.client import LLMClient
from bot.routes.doc_qa.doc_answer_synthesizer_prompt_loader import (
    SynthesizerPromptInput,
    SynthesizerPromptLoader,
)


class AnswerSynthesizer:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_loader: SynthesizerPromptLoader
    ) -> None:
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader

    def synthesize(
        self,
        *,
        question: str,
        chunks_for_synthesis: list[str]
    ) -> str:
        if not chunks_for_synthesis:
            return "I'm sorry, I couldn't find any relevant information in the documents."
        
        system_prompt = self._prompt_loader.load_system_instructions()
        user_prompt = self._prompt_loader.build_user_prompt(input=SynthesizerPromptInput(question=question, chunks=chunks_for_synthesis))
        
        llm_response = self._llm_client.generate(
            prompt=user_prompt,
            system_instructions=system_prompt,
        )
        return llm_response