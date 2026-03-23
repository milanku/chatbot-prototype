from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # LLM configuration
    OPENAI_LLM_PROVIDER: str
    OPENAI_LLM_MODEL: str
    OPENAI_API_KEY: str
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"

    # Paths
    DOCS_PATH: str = "data/docs"
    TRACES_PATH: str = "traces"
    
settings = Settings()