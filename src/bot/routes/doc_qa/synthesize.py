from dataclasses import dataclass
from pathlib import Path

from bot.llm.openai_client import OpenAIClient
from bot.models.repository import DocHit


@dataclass(frozen=True)
class SynthesizePromptInput:
    message: str
    
def load_synthesize_doc_answer_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_synthesize_doc_answer_system_prompt(template: str)-> str:
    return template

def build_synthesize_doc_answer_user_prompt(*, question: str,hits: list[DocHit]) -> str:
    if not hits:
        return "I'm sorry, I couldn't find any relevant information in the documents."
    
    context_block = "\n\n".join(hit.content for hit in hits)
    return (
        f"Answer the user's question based on the following relevant information retrieved from the documents:\n\n"
        f"Question: {question}\n\n"
        f"Retrieved information: \n{context_block}\n\n"
    )

def synthesize_doc_answer(*, llm_client: OpenAIClient, question: str, hits: list[DocHit]) -> str:
    prompt_template = load_synthesize_doc_answer_instructions(Path("src/bot/prompts/synthesize_doc_answer_instructions.txt"))
    system_prompt = build_synthesize_doc_answer_system_prompt(
        template=prompt_template
    )
    user_prompt = build_synthesize_doc_answer_user_prompt(question=question,hits=hits)
    
    llm_response = llm_client.generate(
        prompt=user_prompt,
        system_instructions=system_prompt,
    )
    return llm_response