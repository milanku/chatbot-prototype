
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptConfig:
    directory: Path
    version: str
    
    @property
    def instructions_file_path(self) -> Path:
        return self.directory / f"{self.version}.txt"
    
@dataclass(frozen=True)
class PromptConfigs:
    router: PromptConfig
    claim_extractor: PromptConfig
    claim_verifier: PromptConfig
    explain_summary_parse: PromptConfig
    timeframe_parser: PromptConfig
    doc_answer_synthesizer: PromptConfig

PROMPT_CONFIGS = PromptConfigs(
    router=PromptConfig(
        directory=Path("src/bot/prompts/router_instructions"),
        version="v001",
    ),
    claim_extractor=PromptConfig(
        directory=Path("src/bot/prompts/claim_extractor_instructions"),
        version="v001",
    ),
    claim_verifier=PromptConfig(
        directory=Path("src/bot/prompts/claim_verificator_instructions"),
        version="v002",
    ),
    explain_summary_parse=PromptConfig(
        directory=Path("src/bot/prompts/explain_summary_parser_instructions"),
        version="v001",
    ),
    timeframe_parser=PromptConfig(
        directory=Path("src/bot/prompts/timeframe_parser_instructions"),
        version="v001",
    ),
    doc_answer_synthesizer=PromptConfig(
        directory=Path("src/bot/prompts/doc_answer_synthesizer_instructions"),
        version="v001",
    ),
)