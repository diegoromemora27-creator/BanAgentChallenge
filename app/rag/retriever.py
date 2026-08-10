"""
Capa de Embeddings y cliente de consulta a Qdrant Cloud.
Maneja la inicialización de la colección y la búsqueda por similitud de cosenos.
"""

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.config import settings

logger = logging.getLogger(__name__)

# Carga del modelo local de embeddings (384 dimensiones, optimizado para español e inglés)
try:
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info("Modelo de embeddings %s cargado correctamente.", settings.EMBEDDING_MODEL_NAME)
except Exception as e:
    logger.error("Error al cargar el modelo de embeddings: %s", e)
    embedding_model = None

# Cliente Qdrant Vector DB
def get_qdrant_client() -> QdrantClient:
    """Retorna una instancia configurada del cliente Qdrant."""
    if settings.QDRANT_URL and settings.QDRANT_API_KEY:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    logger.warning("QDRANT_URL o QDRANT_API_KEY no configuradas. Usando Qdrant en memoria (:memory:).")
    return QdrantClient(location=":memory:")

qdrant_client = get_qdrant_client()


def ensure_collection_exists():
    """Garantiza la existencia de la colección en Qdrant."""
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if settings.QDRANT_COLLECTION_NAME not in collections:
            logger.info("Creando colección '%s' en Qdrant...", settings.QDRANT_COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
            )
    except Exception as err:
        logger.error("Error verificando/creando la colección en Qdrant: %s", err)


def retrieve_cv_context(query: str, top_k: int = 4, tipo: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda semántica de información relevante del CV en Qdrant.

    Args:
        query: Pregunta o texto de consulta.
        top_k: Número máximo de resultados a recuperar.
        tipo: Filtro opcional por tipo de chunk ("experiencia", "proyecto", "skills").

    Returns:
        Lista de diccionarios con el texto recuperado, score de similitud y metadatos.
    """
    ensure_collection_exists()

    if not embedding_model:
        logger.error("Modelo de embeddings no disponible.")
        return []

    # Generación de vector de la consulta
    query_vector = embedding_model.encode(query).tolist()

    query_filter = None
    if tipo:
        query_filter = Filter(must=[FieldCondition(key="tipo", match=MatchValue(value=tipo))])

    try:
        results = qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=settings.SCORE_THRESHOLD,
        )

        extracted_context = []
        for r in results:
            extracted_context.append({
                "texto": r.payload.get("texto", ""),
                "score": r.score,
                "metadata": r.payload
            })
        return extracted_context

    except Exception as exc:
        logger.error("Error durante la búsqueda vectorial en Qdrant: %s", exc)
        return []
