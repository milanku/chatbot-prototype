from dataclasses import dataclass

from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class TXExplainParserPromptInput:
    message: str
    
class TXExplainParserPromptLoader(PromptLoader[TXExplainParserPromptInput]):
    def build_user_prompt(self, input: TXExplainParserPromptInput) -> str:
        return (
            "Parse the user's transaction-explain reference.\n\n"
            f"User Message: {input.message!r}\n"
        )