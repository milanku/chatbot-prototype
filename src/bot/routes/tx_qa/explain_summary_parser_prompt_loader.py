from dataclasses import dataclass

from bot.models.prompts import PromptLoader


@dataclass(frozen=True)
class ExplainTxSummaryParserPromptInput:
    message: str
    
class ExplainTxSummaryParserPromptLoader(PromptLoader[ExplainTxSummaryParserPromptInput]):
    def build_user_prompt(self, input: ExplainTxSummaryParserPromptInput) -> str:
        return (
            "Parse the user's transaction-explain reference.\n\n"
            f"User Message: {input.message!r}\n"
        )