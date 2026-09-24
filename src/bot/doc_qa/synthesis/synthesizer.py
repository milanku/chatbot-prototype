from bot.doc_qa.synthesis.synthesizer_prompt_loader import (
    SynthesizerPromptInput,
    SynthesizerPromptLoader,
)
from bot.llm.client import LLMClient


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
        system_prompt = self._prompt_loader.load_system_instructions()
        user_prompt = self._prompt_loader.build_user_prompt(
            input=SynthesizerPromptInput(
                question=question,
                chunks=chunks_for_synthesis
            )
        )
        
        synthesized_answer = self._llm_client.generate(
            prompt=user_prompt,
            system_instructions=system_prompt,
        )
        
        return synthesized_answer