
import argparse

from bot.config.prompts_config import PromptConfig
from bot.config.settings import Settings
from bot.llm import openai_client
from bot.logging import setup_logging
from bot.routes.doc_qa.verifier.verifier_prompt_loader import (
    ClaimVerifierPromptLoader,
)
from prompt_evals.claim_verifier.config import (
    CASES_PATH,
    EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG,
    RESULTS_DIR,
)
from prompt_evals.claim_verifier.evaluator import evaluate_claim_verifier_batch
from prompt_evals.claim_verifier.loader import load_claim_verifier_test_batches
from prompt_evals.claim_verifier.persistor import save_results
from prompt_evals.claim_verifier.print import print_results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
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

    prompt_config = PromptConfig(
        directory=EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG.directory,
        version=args.prompt or EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG.version
    )
    prompt_loader = ClaimVerifierPromptLoader(
        prompt_config=prompt_config
    )

    batches = load_claim_verifier_test_batches(file_path=CASES_PATH)

    evaluation_results = [
        evaluate_claim_verifier_batch(
            llm_client=llm_client,
            batch=batch,
            prompt_loader=prompt_loader
        ) for batch in batches
    ]

    print_results(
        prompt_config=prompt_config,
        batch_results=evaluation_results
    )
            
    save_results(
        prompt_config=prompt_config,
        results_dir=RESULTS_DIR,
        results=evaluation_results
    )       


if __name__ == "__main__":
    main()