from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # LLM configuration
    OPENAI_LLM_PROVIDER: str = "openai"
    OPENAI_LLM_MODEL: str = "gpt-4.1-mini"
    OPENAI_API_KEY: SecretStr = SecretStr("sk-proj-XXX") # Gets overridden by environment variable OPENAI_API_KEY
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "jinaai/jina-embeddings-v3"
    #EMBEDDINGS_MODEL: str = "tencent/WeMM-Embedding-2B"
    CHUNKING_VERSION: str = "v1.0.0"
    
    # Paths
    DOCS_PATH: str = "data/docs"
    EMBEDDINGS_PATH: str = "data/embeddings"
    EMBEDDINGS_MANIFEST_PATH: str = "data/embeddings/manifest.json"
    TRANSACTIONS_MOCK_PATH: str = "data/mocks/transactions_mock.json"
    TRACES_PATH: str = "traces"
    
settings = Settings()