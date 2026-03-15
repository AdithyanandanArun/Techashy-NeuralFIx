from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""

    # Groq
    groq_api_key: str = ""

    # Database
    database_url: str = "postgresql://netfix:netfix123@localhost:5432/netfixai"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # RAG
    docs_path: str = "./docs"
    vector_store_path: str = "./vector_store"

    # App
    app_name: str = "NetFixAI"
    debug: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
