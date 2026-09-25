import json
from datetime import datetime
from pathlib import Path

from prompt_evals.claim_verifier.models import ClaimVerifierBatchEvaluationResult

from bot.config.prompts_config import PromptConfig


def save_results(
    *,
    prompt_config: PromptConfig,
    results_dir: Path,
    results: list[ClaimVerifierBatchEvaluationResult],
    today: datetime,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    file_path = results_dir / f"{prompt_config.version}_{today.strftime('%Y-%m-%d_%H:%M:%S')}.json"
    with open(file_path, "w") as f:
        json.dump([result.model_dump() for result in results], f)
