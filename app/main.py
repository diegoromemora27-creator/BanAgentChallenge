"""
Aplicación principal FastAPI para el Agente Conversacional de CV.
Mantiene la configuración global, middlewares de CORS, Rate Limiter e inyecta los routers modulares de la API.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.rag.ingest import ingest_cv

from app.api.meta import router as meta_router
from app.api.chat import router as chat_router
from app.api.cv import router as cv_router

# Configuración del Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración del Limiter para protección de cuota (60 peticiones/minuto por IP)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para el Agente Conversacional de CV basado en RAG estricto, Qdrant Cloud y estándar Open Responses."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuración de CORS universal para permitir peticiones desde navegadores y la plataforma del Hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://hackathon-2024.com",
        "http://hackathon-2024.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
def auto_ingest_default_cv():
    """Ingesta automáticamente el CV de Diego Romero Mora (data/cv_sample.txt) si Qdrant está vacío."""
    try:
        from app.rag.retriever import qdrant_client, ensure_collection_exists
        ensure_collection_exists()
        info = qdrant_client.get_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
        if info.points_count == 0:
            logger.info("Qdrant está vacío. Auto-ingestionando CV de Diego Romero Mora desde data/cv_sample.txt...")
            with open("data/cv_sample.txt", "r", encoding="utf-8") as f:
                pasted_text = f.read()
            ingest_cv(pasted_text=pasted_text)
            logger.info("CV de Diego Romero Mora indexado exitosamente en Qdrant.")
        else:
            logger.info("Qdrant ya contiene %d puntos indexados para el CV.", info.points_count)
    except Exception as exc:
        logger.warning("No se pudo verificar o autoingestar el CV en startup: %s", exc)


# Inclusión de Routers Modulares
app.include_router(meta_router)
app.include_router(chat_router)
app.include_router(cv_router)
