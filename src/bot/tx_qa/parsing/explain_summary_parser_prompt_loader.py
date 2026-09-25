from dataclasses import dataclass

from bot.prompts.models import PromptLoader


@dataclass(frozen=True)
class ExplainTxSummaryParserPromptInput:
    message: str


class ExplainTxSummaryParserPromptLoader(PromptLoader[ExplainTxSummaryParserPromptInput]):
    def build_user_prompt(self, input: ExplainTxSummaryParserPromptInput) -> str:
        return (
            f"Parse the user's transaction-explain reference.\n\nUser Message: {input.message!r}\n"
        )
