from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl
from typing import List, Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "SkyGuard AI"
    API_V1_STR: str = "/api"
    # Postgres database connection string format: postgresql://user:password@server/db
    # We default to sqlite for local testing if not provided
    DATABASE_URL: str = "sqlite:///./skyguard.db"
    
    # CORS setup
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]
    
    # ML Models Path
    MODEL_DIR: str = "ml/models/saved"
    DATA_DIR: str = "data/processed"
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
