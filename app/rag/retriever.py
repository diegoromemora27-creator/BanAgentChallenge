"""
Capa de Embeddings por API y cliente de consulta a Qdrant Cloud.
Maneja la inicialización de la colección y la búsqueda por similitud de cosenos sin cargar modelos pesados en RAM.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from app.config import settings

logger = logging.getLogger(__name__)

# Cliente de Hugging Face Router para embeddings vía API (ligero, consume 0 MB de RAM)
hf_embedding_client = OpenAI(
    api_key=settings.HF_TOKEN or "dummy_token_if_empty",
    base_url="https://router.huggingface.co/v1"
)

def get_embedding(text_or_texts: str | List[str]) -> List[List[float]]:
    """
    Genera vectores de embeddings realizando llamadas a la API de OpenAI o Hugging Face Inference Router.
    No requiere PyTorch ni modelos pesados en la RAM local.
    """
    inputs = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
    
    # Intentamos primero generar con el cliente de OpenAI / Hugging Face Router
    try:
        response = hf_embedding_client.embeddings.create(
            model="BAAI/bge-small-en-v1.5",
            input=inputs
        )
        return [data.embedding for data in response.data]
    except Exception as exc:
        logger.warning("No se pudo obtener embedding vía HF Router API (%s). Generando vector con fallback...", exc)
        # Vector nulo con la dimensión configurada para evitar bloqueos
        return [[0.01] * settings.EMBEDDING_VECTOR_SIZE for _ in inputs]


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

    # Generación de vector de la consulta vía API
    query_vectors = get_embedding(query)
    if not query_vectors:
        return []
    
    query_vector = query_vectors[0]

    query_filter = None
    if tipo:
        query_filter = Filter(must=[FieldCondition(key="tipo", match=MatchValue(value=tipo))])

    try:
        # En qdrant-client >= 1.10.0 se utiliza query_points o search
        if hasattr(qdrant_client, "query_points"):
            response = qdrant_client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            results = response.points
        else:
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
