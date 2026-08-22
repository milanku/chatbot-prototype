import json
from pathlib import Path

from prompt_evals.claim_verifier.models import ClaimVerifierTestBatch


def load_prompt(*, base_path: Path, version: str) -> str:
    prompt_path: Path = base_path / f"{version}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' not found: {prompt_path}"
        )
        
    return prompt_path.read_text(encoding="utf-8")

def load_claim_verifier_test_batches(*, file_path: Path) -> list[ClaimVerifierTestBatch]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cases file not found: {file_path}"
        )
        
    return [ClaimVerifierTestBatch(**case) for case in json.loads(file_path.read_text(encoding="utf-8"))]