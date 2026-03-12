from hmac import new

from attr import dataclass

from bot.models.repository import DocChunk, DocRepository
from bot.recipes.doc_qa.score import score_chunks
from bot.recipes.doc_qa.tokenizer import tokenize


@dataclass
class DocStore(DocRepository):
    chunks: list[DocChunk]
    
    def search(self, query: str, *, top_k: int = 5) -> list[DocChunk]:
        return self.chunks[:top_k]
    
    def split_into_chunks(self, file_name: str, content: str) -> list[DocChunk]:
        new_chunks = []
        #splits markdown content into chunks by titles, and assigns a chunk_id to each chunk
        chunk_id = 0
        breadcrumbs_stack = []
        for line in content.splitlines():
            normalized_line = line.strip()
            if(normalized_line.startswith("##")):
                breadcrumbs_stack.append(normalized_line.replace("##", "", 1).strip())
            elif(normalized_line.startswith("#")):
                breadcrumbs_stack.append(normalized_line.replace("#", "", 1).strip())
                chunk_content = "\n".join(breadcrumbs_stack) + "\n" + normalized_line
                
                new_chunks.append(DocChunk(file_name=file_name, content=chunk_content, chunk_id=chunk_id))
            new_chunks.append(DocChunk(file_name=file_name, content=normalized_line, chunk_id=chunk_id))
        return new_chunks
    
    def get_top_k_chunks(self, query, *, top_k = 5):
        query_tokens = tokenize(query)
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk.content)
            chunk.score = score_chunks(query_tokens, chunk_tokens)
        sorted_chunks = sorted(self.chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:top_k]