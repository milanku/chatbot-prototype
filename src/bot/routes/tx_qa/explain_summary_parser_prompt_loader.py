from dataclasses import dataclass

from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class TXExplainSummaryParserPromptInput:
    message: str
    
class TXExplainSummaryParserPromptLoader(PromptLoader[TXExplainSummaryParserPromptInput]):
    def build_user_prompt(self, input: TXExplainSummaryParserPromptInput) -> str:
        return (
            "Parse the user's transaction-explain reference.\n\n"
            f"User Message: {input.message!r}\n"
        )