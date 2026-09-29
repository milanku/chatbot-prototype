from pathlib import Path

from bot.config.prompts_config import PromptConfig

CASES_PATH = Path("prompt_evals/claim_verifier/cases.json")
OUTPUT_DIR = Path("prompt_evals/claim_verifier/output")

EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG = PromptConfig(
    package="bot",
    directory=Path("prompts/doc_qa/claim_verificator_instructions"),
    version="v003",
)
