from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DateParserPromptInput:
    message: str
    
def load_date_parser_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_date_parser_system_prompt(template: str)-> str:
    return template

def build_date_parser_user_prompt(input: DateParserPromptInput) -> str:
    return f"""Extract the timeframe information from the following user message.
    Message: \"{input.message}\""""