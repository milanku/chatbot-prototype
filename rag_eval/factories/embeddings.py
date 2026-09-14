from sentence_transformers import SentenceTransformer

from bot.routes.doc_qa.local_embeddings import LocalEmbeddings


def get_embeddings(embeddings_model:str) -> LocalEmbeddings:
    if(embeddings_model.startswith("jina") or embeddings_model.startswith("qwen")):
        embeddings = LocalEmbeddings(
            SentenceTransformer(
                embeddings_model,
                trust_remote_code=True,
            )
        )
        return embeddings
    raise ValueError(f"Unsupported embeddings model: {embeddings_model}")