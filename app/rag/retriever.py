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

import time
import requests

def get_embedding(text_or_texts: str | List[str]) -> List[List[float]]:
    """
    Genera vectores de embeddings semánticos reales consumiendo la API oficial de Hugging Face.
    Utiliza el modelo oficial 'sentence-transformers/all-MiniLM-L6-v2' (dimensión 384).
    """
    inputs = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
    
    # Nuevo endpoint oficial de Hugging Face Router para Feature Extraction
    api_urls = [
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
    ]
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"} if settings.HF_TOKEN else {}

    for api_url in api_urls:
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": inputs, "options": {"wait_for_model": True}},
                timeout=8
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
                        sentence_embeddings = []
                        for sent in data:
                            dim = len(sent[0])
                            avg_vec = [sum(sent[t][d] for t in range(len(sent))) / len(sent) for d in range(dim)]
                            sentence_embeddings.append(avg_vec)
                        return sentence_embeddings
                    elif isinstance(data[0], list) and isinstance(data[0][0], (float, int)):
                        return data
            logger.warning("API de Hugging Face respondió status %d en %s: %s", response.status_code, api_url, response.text[:150])
        except Exception as exc:
            logger.warning("Fallo al obtener embedding de Hugging Face API en %s (%s)", api_url, exc)

    logger.error("No se pudo obtener embeddings semánticos de Hugging Face.")
    return []


# Cliente Qdrant Vector DB
def get_qdrant_client() -> QdrantClient:
    """Retorna una instancia configurada del cliente Qdrant."""
    if settings.QDRANT_URL and settings.QDRANT_API_KEY:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    logger.warning("QDRANT_URL o QDRANT_API_KEY no configuradas. Usando Qdrant en memoria (:memory:).")
    return QdrantClient(location=":memory:")

qdrant_client = get_qdrant_client()


from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType

def ensure_collection_exists():
    """Garantiza la existencia de la colección en Qdrant e índice de metadatos."""
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
        
        # Garantiza los índices de payload requeridos por Qdrant Cloud para borrados y filtros
        try:
            qdrant_client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="cv_version",
                field_schema=PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="tipo",
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass # Si el índice ya existe, ignora la advertencia

    except Exception as err:
        logger.error("Error verificando/creando la colección en Qdrant: %s", err)


import re

def clean_query_for_embedding(query: str) -> str:
    """Remueve saludos e interjecciones comunes para no distorsionar el vector semántico."""
    cleaned = re.sub(r'^(buenas\s+noches|buenas\s+tardes|buenos\s+días|hola|buenas|saludos|hey|hi|hello)[,!\s]*', '', query, flags=re.IGNORECASE).strip()
    return cleaned if len(cleaned) > 2 else query



def retrieve_cv_context(query: str, top_k: int = 4, tipo: Optional[str | List[str]] = None, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda semántica de información relevante del CV en Qdrant.

    Args:
        query: Pregunta o texto de consulta.
        top_k: Número máximo de resultados a recuperar.
        tipo: Filtro opcional por tipo o lista de tipos de chunk.
        score_threshold: Umbral mínimo de similitud Cosine.

    Returns:
        Lista de diccionarios con el texto recuperado, score de similitud y metadatos.
    """
    ensure_collection_exists()

    # Limpieza de saludos en el texto del query únicamente para el cálculo de embeddings
    clean_query = clean_query_for_embedding(query)

    # Generación de vector de la consulta vía API
    query_vectors = get_embedding(clean_query)
    if not query_vectors:
        return []
    
    query_vector = query_vectors[0]

    query_filter = None
    if tipo:
        if isinstance(tipo, list):
            query_filter = Filter(must=[FieldCondition(key="tipo", match=MatchAny(any=tipo))])
        else:
            query_filter = Filter(must=[FieldCondition(key="tipo", match=MatchValue(value=tipo))])

    try:
        # En qdrant-client >= 1.10.0 se utiliza query_points o search
        if hasattr(qdrant_client, "query_points"):
            response = qdrant_client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            results = response.points
        else:
            results = qdrant_client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )

        extracted_context = []
        for r in results:
            # Umbral de similitud para eliminar vectores ruidosos
            if r.score >= score_threshold:
                extracted_context.append({
                    "id": str(getattr(r, "id", "")),
                    "texto": r.payload.get("texto", ""),
                    "score": r.score,
                    "metadata": r.payload
                })
        return extracted_context

    except Exception as exc:
        logger.error("Error durante la búsqueda vectorial en Qdrant: %s", exc)
        return []
