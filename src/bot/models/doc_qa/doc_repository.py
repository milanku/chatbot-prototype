from bot.models.doc_qa.retrieval import DocHit


class DocRepository():
    def search(self, query: str, *, top_k: int = 5) -> list[str]: ...
    
    def get_top_k_chunks(self, query: str, *, top_k: int = 5) -> list[DocHit]: ...