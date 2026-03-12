from hmac import new

from attr import dataclass
from zipp import Path

from bot.models.repository import DocChunk, DocRepository
from bot.recipes.doc_qa.score import score_chunks
from bot.recipes.doc_qa.tokenizer import tokenize


@dataclass
class MockBankDocStore(DocRepository):
    chunks: list[DocChunk]
    
    @classmethod   
    def from_files(cls, file_paths: list[Path]) -> "MockBankDocStore":
        all_chunks = []
        for file_path in file_paths:
            content = file_path.read_text(encoding="utf-8")
            chunks = cls.split_into_chunks(file_path.name, content)
            all_chunks.extend(chunks)
        return cls(chunks=all_chunks)
    
    def search(self, query: str, *, top_k: int = 5) -> list[DocChunk]:
        return self.chunks[:top_k]
    
    @staticmethod
    def split_into_chunks(file_name: str, content: str) -> list[DocChunk]:
        new_chunks = []
        # Splits markdown content into chunks by newlines, and assigns a chunk_id to each chunk
        title_stack = []
        chunk_id = 0
        current_chunk_lines = []
        
        def flush_chunk():
            nonlocal chunk_id
            if current_chunk_lines:
                # Prepend the current titles stack to the chunk content
                chunk_content = "\n".join(title_stack) + "\n" + "\n".join(current_chunk_lines)
                new_chunks.append(DocChunk(file_name=file_name, content=chunk_content, chunk_id=chunk_id))
                chunk_id += 1
                current_chunk_lines.clear()

        for line in content.splitlines():
            normalized_line = line.strip()
            if not normalized_line:
                continue  # Skip empty lines

            if normalized_line.startswith("#"):
                # Determine the level of the heading
                level = len(normalized_line) - len(normalized_line.lstrip("#"))
                title = normalized_line.lstrip("#").strip()

                # Remove titles from the stack that are deeper than or equal to the current level
                while len(title_stack) >= level:
                    title_stack.pop()

                # Add the current title to the stack
                title_stack.append(title)

                # If this is a higher-level title and there are subheadings, don't create a chunk
                if level > 1:
                    flush_chunk()  # Close the current chunk before starting a new section
            else:
                # Add the line to the current chunk
                current_chunk_lines.append(normalized_line)

        # Flush the last chunk
        flush_chunk()
        return new_chunks
    
    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocChunk]:
        query_tokens = tokenize(query)
        chunk_scores = []
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk.content)
            chunk_score = score_chunks(query_tokens, chunk_tokens)
            chunk_scores.append((chunk, chunk_score))
        sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in sorted_chunks[:top_k]]