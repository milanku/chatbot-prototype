from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TXExplainParserPromptInput:
    message: str
    
def load_tx_explain_parser_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_tx_explain_parser_system_prompt(template: str)-> str:
    return template

def build_tx_explain_parser_user_prompt(input: TXExplainParserPromptInput) -> str:
    return (
        "Parse the user's transaction-explain reference.\n\n"
        f"User Message: {input.message!r}\n"
    )