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
    
    # Persistencia Externa con Supabase PostgreSQL (Checkpointer)
    DATABASE_URL: str = Field(default="", description="URI de conexión a Supabase PostgreSQL para persistencia de estado")
    
    # Observabilidad Avanzada con Langfuse Cloud (Opcional)
    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Public Key para Langfuse Cloud Tracing")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Secret Key para Langfuse Cloud Tracing")
    LANGFUSE_HOST: str = Field(default="https://us.cloud.langfuse.com", description="Host de Langfuse Cloud")
    LANGFUSE_BASE_URL: str = Field(default="", description="Alias para Host de Langfuse Cloud")

    # Token de Seguridad para Protección del Endpoint /metrics en Grafana Cloud
    METRICS_TOKEN: str = Field(default="banorte_metrics_secret_token_2026", description="Token Bearer para autenticar Grafana Cloud scraper")
    
    # RAG Settings (Embeddings vía API sin carga local en RAM)
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
