from bot.doc_qa.indexing.models import Chunker, DocChunk, DocReference


class ContextualChunker(Chunker):
    def split(self, text: str, file_name: str) -> list[DocChunk]:
        """
        Splits markdown into chunks by paragraphs. Prepends title "breadcrumb" to the chunk.

        Args:
            file_name (str): file_name
            text (str): markdown content

        Returns:
            list[DocChunk]: List of retrieved chunks
        """

        new_chunks: list[DocChunk] = []
        # Splits markdown content into chunks by newlines, and assigns a chunk_id to each chunk
        title_stack: list[str] = []
        chunk_index = 0
        current_chunk_lines: list[str] = []

        def flush_chunk() -> None:
            nonlocal chunk_index
            if current_chunk_lines:
                # Prepend the current titles stack to the chunk content
                chunk_content = "\n".join(title_stack) + "\n" + "\n".join(current_chunk_lines)
                new_chunks.append(
                    DocChunk(
                        doc_reference=DocReference(
                            file_name=file_name, heading_path=title_stack.copy()
                        ),
                        content=chunk_content,
                        chunk_id=f"{file_name}_{chunk_index}",
                    )
                )
                chunk_index += 1
                current_chunk_lines.clear()

        for line in text.splitlines():
            normalized_line = line.strip()
            if not normalized_line:
                continue  # Skip empty lines

            if normalized_line.startswith("#"):
                # Determine the level of the heading (number of '#' characters)
                level = len(normalized_line) - len(normalized_line.lstrip("#"))
                title = normalized_line.lstrip("#").strip()

                # Found a new heading, flush the current chunk (if it has content)
                flush_chunk()

                # Remove titles from the stack that are deeper than or equal to the current level
                while len(title_stack) >= level:
                    title_stack.pop()

                # Add the current title to the stack
                title_stack.append(title)
            else:
                # Add the line to the current chunk
                current_chunk_lines.append(normalized_line)

        flush_chunk()
        return new_chunks
