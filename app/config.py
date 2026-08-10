import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuración de la aplicación usando Pydantic Settings.
    Carga variables desde el entorno o archivo .env.
    """
    PROJECT_NAME: str = "CV Agent API"
    VERSION: str = "1.0.0"
    
    # Proveedores de LLM y Vector DB
    GROQ_API_KEY: str = Field(default="", description="API Key para Groq Cloud")
    HF_TOKEN: str = Field(default="", description="Token de acceso para Hugging Face Inference")
    QDRANT_URL: str = Field(default="", description="URL de Qdrant Cloud o local")
    QDRANT_API_KEY: str = Field(default="", description="API Key para Qdrant Cloud")
    
    # RAG Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_VECTOR_SIZE: int = 384
    QDRANT_COLLECTION_NAME: str = "cv_chunks"
    SCORE_THRESHOLD: float = 0.35
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

from pydantic import Field

settings = Settings()
