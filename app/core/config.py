from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Multi-Agent AI Autonomous Research Assistant"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True

    API_V1_PREFIX: str = "/api/v1"

    OPENAI_API_KEY: str = ""
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

    DATASET_DIR: str = "../datasets"
    MODEL_DIR: str = "../models"
    REPORT_DIR: str = "../reports"
    LOG_DIR: str = "../logs"

    MAX_CONCURRENT_JOBS: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()