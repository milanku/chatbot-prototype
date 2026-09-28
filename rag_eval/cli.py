import argparse
from pathlib import Path

from dotenv import load_dotenv

from bot.logging import setup_logging
from rag_eval.artifacts.artifact_step_executor import ArtifactStepExecutor
from rag_eval.artifacts.artifact_store import ArtifactStore
from rag_eval.artifacts.local_artifact_store import LocalArtifactStore
from rag_eval.config import (
    create_pipeline_config,
)
from rag_eval.pipeline.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    load_dotenv()

    config = create_pipeline_config()
    artifact_store: ArtifactStore = LocalArtifactStore(
        base_dir_path=Path("rag_eval/data/artifacts"),
    )
    runner = ArtifactStepExecutor(
        artifact_store=artifact_store,
    )

    pipeline = EvaluationPipeline(config=config, artifact_store=artifact_store, runner=runner)
    pipeline.run()


if __name__ == "__main__":
    main()
