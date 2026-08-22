
import argparse
from pathlib import Path

from bot.config import Settings
from bot.llm import openai_client
from bot.logging import setup_logging
from prompt_evals.claim_verifier.evaluator import evaluate_claim_verifier_batch
from prompt_evals.claim_verifier.loader import load_claim_verifier_test_batches
from prompt_evals.claim_verifier.persistor import save_results
from prompt_evals.claim_verifier.print import print_results

DEFAULT_VERSION = "v002"
PROMPTS_DIR = Path("src/bot/prompts/claim_verification_instructions")
CASES_PATH = Path("prompt_evals/claim_verifier/cases.json")
RESULTS_DIR = Path("prompt_evals/claim_verifier/results")                  

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        default=DEFAULT_VERSION,
        help="Prompt version, e.g. v002",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    settings = Settings()

    llm_client = openai_client.OpenAIClient(
        api_key=settings.OPENAI_API_KEY,
    )

    batches = load_claim_verifier_test_batches(file_path=CASES_PATH)

    evaluation_results = [
        evaluate_claim_verifier_batch(
            llm_client=llm_client,
            batch=batch,
            prompt_version=args.prompt
        ) for batch in batches
    ]

    print_results(
        prompt_version=args.prompt,
        batch_results=evaluation_results
    )
            
    save_results(
        prompt_version=args.prompt,
        results_dir=RESULTS_DIR,
        results=evaluation_results
    )       


if __name__ == "__main__":
    main()