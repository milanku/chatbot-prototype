import argparse
from datetime import datetime

from dotenv import load_dotenv
from prompt_evals.claim_verifier.config import (
    CASES_PATH,
    EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG,
    OUTPUT_DIR,
)
from prompt_evals.claim_verifier.evaluator import ClaimVerifierEvaluator
from prompt_evals.claim_verifier.loader import load_claim_verifier_test_batches
from prompt_evals.claim_verifier.results.persistor import save_results
from prompt_evals.claim_verifier.results.print import print_results

from bot.config.llm import OpenAILLMConfig
from bot.doc_qa.verification.verifier import ClaimsVerifier
from bot.doc_qa.verification.verifier_prompt_loader import ClaimVerifierPromptLoader
from bot.llm.factory import create_llm
from bot.logging import setup_logging


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
    load_dotenv()

    llm_client = create_llm(config=OpenAILLMConfig(model="gpt-4.1-mini"))
    prompt_loader = ClaimVerifierPromptLoader(prompt_config=EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG)

    verifier = ClaimsVerifier(llm_client=llm_client, claim_verifier_prompt_loader=prompt_loader)
    evaluator = ClaimVerifierEvaluator(llm_client=llm_client, verifier=verifier)

    # Load test batches, evaluate, print and save results
    batches = load_claim_verifier_test_batches(file_path=CASES_PATH)
    evaluation_results = [evaluator.evaluate_claim_verifier_batch(batch=batch) for batch in batches]
    print_results(
        prompt_config=EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG, batch_results=evaluation_results
    )
    save_results(
        prompt_config=EVALUATOR_CLAIM_VERIFIER_PROMPT_CONFIG,
        results_dir=OUTPUT_DIR,
        results=evaluation_results,
        today=datetime.now(),
    )


if __name__ == "__main__":
    main()
