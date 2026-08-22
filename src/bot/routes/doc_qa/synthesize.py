
from bot.llm.client import LLMClient
from bot.models.doc_qa.retrieval import DocHit
from bot.routes.doc_qa.doc_answer_synthesizer_prompt_loader import (
    DocAnswerSynthesizerPromptLoader,
    DocSynthesizerPromptInput,
)


def synthesize_doc_answer(
    *,
    llm_client: LLMClient,
    question: str,
    hits: list[DocHit],
    prompt_loader: DocAnswerSynthesizerPromptLoader
) -> str:
    # TODO: Consider moving to more appropriate place
    if not hits:
        return "I'm sorry, I couldn't find any relevant information in the documents."
    
    system_prompt = prompt_loader.load_system_instructions()
    user_prompt = prompt_loader.build_user_prompt(input=DocSynthesizerPromptInput(question=question, hits=hits))
    
    llm_response = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt,
    )
    return llm_response