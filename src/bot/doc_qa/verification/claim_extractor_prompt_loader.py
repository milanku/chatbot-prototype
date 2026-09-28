from dataclasses import dataclass

from bot.prompts.models import PromptLoader


@dataclass(frozen=True)
class ClaimExtractorPromptInput:
    text: str


class ClaimExtractorPromptLoader(PromptLoader[ClaimExtractorPromptInput]):
    def build_user_prompt(self, input: ClaimExtractorPromptInput) -> str:
        return (
            "Extract the independently verifiable claims from the following text.\n\n"
            f"Text:\n\n{input.text}\n"
        )
