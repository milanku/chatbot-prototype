from bot.routes.doc_qa.chunker import split_md_to_chunks_by_paragraphs
from bot.routes.doc_qa.embeddings_store import Chunker


def get_chunker(chunker_version: str) -> Chunker:
    if chunker_version == "paragraph_with_breadcrumb_chunker_v01":
        return split_md_to_chunks_by_paragraphs
    raise ValueError(f"Unsupported chunker version: {chunker_version}")