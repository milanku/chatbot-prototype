from pathlib import Path

from bot.config.prompts_config import PromptConfig

PROMPTS_DIR = Path("src/bot/prompts/claim_verification_instructions")
CASES_PATH = Path("prompt_evals/claim_verifier/cases.json")
RESULTS_DIR = Path("prompt_evals/claim_verifier/results")   

EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG = PromptConfig(
    directory=Path("src/bot/prompts/claim_verification_instructions"),
    version="v002",
)