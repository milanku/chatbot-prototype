from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    docs_dir_path: Path = Path("data/docs")
    transactions_mock_file_path: Path = Path("data/mocks/transactions_mock_jan2025_sep2026.json")
