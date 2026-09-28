from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptConfig:
    directory: Path
    version: str
    package: str = "bot"

    @property
    def instructions_file_path(self) -> Path:
        return self.directory / f"{self.version}.txt"


@dataclass(frozen=True)
class PromptConfigs:
    router: PromptConfig
    chunk_judge: PromptConfig
    doc_answer_synthesizer: PromptConfig
    claim_extractor: PromptConfig
    claim_verifier: PromptConfig
    explain_tx_summary_parser: PromptConfig
    timeframe_parser: PromptConfig


PROMPT_CONFIGS = PromptConfigs(
    router=PromptConfig(
        directory=Path("prompts/router_instructions"),
        version="v001",
    ),
    chunk_judge=PromptConfig(
        directory=Path("prompts/doc_qa/chunk_judge"),
        version="two-way-judge-v1",
    ),
    doc_answer_synthesizer=PromptConfig(
        directory=Path("prompts/doc_qa/doc_answer_synthesizer_instructions"),
        version="v001",
    ),
    claim_extractor=PromptConfig(
        directory=Path("prompts/doc_qa/claim_extractor_instructions"),
        version="v002",
    ),
    claim_verifier=PromptConfig(
        directory=Path("prompts/doc_qa/claim_verificator_instructions"),
        version="v003",
    ),
    explain_tx_summary_parser=PromptConfig(
        directory=Path("prompts/tx_qa/explain_tx_summary_parser_instructions"),
        version="v001",
    ),
    timeframe_parser=PromptConfig(
        directory=Path("prompts/tx_qa/timeframe_parser_instructions"),
        version="v001",
    ),
)
