# load env file
from pydantic import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str
    APP_DESCRIPTION: str
    APP_VERSION: str
    HK_MODELS_CHECKPOINT_PATH: str
    US_MODELS_CHECKPOINT_PATH: str
    EVALUATION_STOCKS_PATH: str
    SCALER_PATH: str
    RAW_DATA_PATH: str
    FRONT_END_URL: str
    API_TIMEOUT_MINUTES: int

    class Config:
        env_file = ".env"

# @lru_cache()
# def get_settings():
#     return Settings

# settings = get_settings()

settings = Settings()