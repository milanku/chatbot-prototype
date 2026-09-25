from pathlib import Path

from pydantic import BaseModel


class MarkdownDocument(BaseModel):
    file_path: Path
    content: str


def read_markdown_docs(docs_dir_path: Path) -> list[MarkdownDocument]:
    md_file_paths = list(docs_dir_path.glob("*.md"))

    documents: list[MarkdownDocument] = []
    for file_path in md_file_paths:
        document_content = file_path.read_text(encoding="utf-8")
        documents.append(MarkdownDocument(file_path=file_path, content=document_content))

    return documents
