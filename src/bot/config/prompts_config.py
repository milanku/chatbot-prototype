
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptConfig:
    directory: Path
    version: str
    
    @property
    def instructions_file_path(self) -> Path:
        return self.directory / f"{self.version}.txt"

CLAIM_VERIFIER_PROMPT_CONFIG = PromptConfig(
    directory=Path("src/bot/prompts/claim_verification_instructions"),
    version="v002",
)