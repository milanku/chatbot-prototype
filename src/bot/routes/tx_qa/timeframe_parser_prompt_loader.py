from dataclasses import dataclass

from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class TimeframeParserPromptInput:
    message: str
    
class TimeframeParserPromptLoader(PromptLoader[TimeframeParserPromptInput]):
    def build_user_prompt(self, input: TimeframeParserPromptInput) -> str:
         return f"""Extract the timeframe information from the following user message.
             Message: \"{input.message}\""""