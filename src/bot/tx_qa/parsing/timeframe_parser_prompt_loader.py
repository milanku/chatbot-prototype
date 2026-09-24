from dataclasses import dataclass

from bot.prompts.models import PromptLoader


@dataclass(frozen=True)
class TimeframeParserPromptInput:
    message: str
    
class TimeframeParserPromptLoader(PromptLoader[TimeframeParserPromptInput]):
    def build_user_prompt(self, input: TimeframeParserPromptInput) -> str:
        return (
            f"Extract the timeframe information from the following user message.\n"
            f"Message: \"{input.message}\""
        )