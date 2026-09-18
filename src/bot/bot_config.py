from pydantic_settings import BaseSettings, SettingsConfigDict


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "jinaai/jina-embeddings-v3"
    #EMBEDDINGS_MODEL: str = "tencent/WeMM-Embedding-2B"
    CHUNKING_VERSION: str = "v1.0.0"
    
    # Paths
    DOCS_PATH: str = "data/docs"
    EMBEDDINGS_PATH: str = "data/embeddings"
    EMBEDDINGS_MANIFEST_PATH: str = "data/embeddings/manifest.json"
    TRANSACTIONS_MOCK_PATH: str = "data/mocks/transactions_mock_jan2025_sep2026.json"
    TRACES_PATH: str = "traces"
    
bot_config = EngineConfig()