import json
from pathlib import Path

from prompt_evals.claim_verifier.models import ClaimVerifierBatchEvaluationResult


def save_results(
    *,
    prompt_version: str,
    results_dir: Path,
    results: list[ClaimVerifierBatchEvaluationResult]
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    file_path = results_dir / f"{prompt_version}.json"
    with open(file_path, "w") as f:
        json.dump([result.model_dump() for result in results], f)