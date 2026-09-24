from pathlib import Path

from bot.config.prompts_config import PromptConfig

PROMPTS_DIR = Path("src/bot/prompts/claim_verificator_instructions")
CASES_PATH = Path("prompt_evals/claim_verifier/cases.json")
OUTPUT_DIR = Path("prompt_evals/claim_verifier/output")   

EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG = PromptConfig(
    directory=Path("src/bot/prompts/claim_verificator_instructions"),
    version="v003",
)