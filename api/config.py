import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "EC Price Predictor API"
    app_version: str = "1.0.0"

    database_url: str = "postgresql://ec_user:ec_pass@localhost:5432/ec_db"
    ura_access_key: str = ""

    model_dir: str = "models"
    secret_key: str = "change-me-in-production"
    admin_token: str = "admin-secret"

    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
