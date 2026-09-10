import argparse

from dateparser.conf import Settings
from pydantic.v1 import BaseSettings
from sentence_transformers import SentenceTransformer

from bot.logging import setup_logging
from bot.routes.doc_qa.doc_store import DocStore
from bot.routes.doc_qa.local_embeddings import LocalEmbeddings


class EvalPipelineSettings(BaseSettings):
    EMBEDDINGS_MODEL: str = "qwen/Qwen3-Embedding-4B"
    EMBEDDINGS_PATH: str = "rag_evals/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setsize",
        type=int,
        help="Set size for the evaluation set, e.g. 100 [questions]",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    
    # 1. Generate embeddings using different embedder than production pipeline
    sentence_transformer = SentenceTransformer(
        "qwen/Qwen3-Embedding-4B",
        trust_remote_code=True,
    )
    qwen_embedder = LocalEmbeddings(sentence_transformer)
    doc_store = DocStore.build_doc_store(
        embedder=qwen_embedder,
        embedding_model=settings.EMBEDDINGS_MODEL,
        embeddings_dir=Path(settings.EMBEDDINGS_PATH),
        md_docs_dir=Path(settings.DOCS_PATH),
        manifest_path=Path(settings.EMBEDDINGS_MANIFEST_PATH),
        chunking_version=settings.CHUNKING_VERSION,
    )
    
    
if __name__ == "__main__":
    main()